import cv2
import numpy as np
import onnxruntime as ort

class ONNXDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]

    def preprocess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0).astype(np.float32)
        img /= 255.0
        return img

    def postprocess(self, output, conf_threshold=0.25, iou_threshold=0.45):
        predictions = np.squeeze(output[0])
        predictions = predictions.T
        
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]
        
        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        results_boxes = np.stack([x1, y1, x2, y2], axis=1)
        nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()
        
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_threshold, iou_threshold)
        
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            return results_boxes[indices], scores[indices], class_ids[indices]
        return [], [], []

    def detect(self, image):
        img_height, img_width = image.shape[:2]
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes, scores, class_ids = self.postprocess(outputs)
        
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= (img_width / self.input_width)
            boxes[:, [1, 3]] *= (img_height / self.input_height)
            
        return boxes, scores, class_ids
