import supervision as sv
import time
import numpy as np

class RetailTracker:
    def __init__(self, line_start=(0, 300), line_end=(640, 300)):
        self.tracker = sv.ByteTrack()
        self.line_zone = sv.LineZone(
            start=sv.Point(line_start[0], line_start[1]), 
            end=sv.Point(line_end[0], line_end[1])
        )
        self.entry_times = {}
        self.dwell_durations = []
        self.active_tracks = set()

    def update(self, boxes, scores, class_ids):
        if len(boxes) == 0:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections(
                xyxy=boxes.astype(np.float32),
                confidence=scores.astype(np.float32),
                class_id=class_ids.astype(np.int32)
            )
        
        detections = self.tracker.update_with_detections(detections)
        self.line_zone.trigger(detections)
        
        current_time = time.time()
        current_track_ids = set()
        
        if detections.tracker_id is not None:
            for track_id in detections.tracker_id:
                current_track_ids.add(track_id)
                if track_id not in self.entry_times:
                    self.entry_times[track_id] = current_time
        
        for track_id in list(self.entry_times.keys()):
            if track_id not in current_track_ids and track_id in self.active_tracks:
                duration = current_time - self.entry_times[track_id]
                self.dwell_durations.append(duration)
                self.entry_times.pop(track_id)
        
        self.active_tracks = current_track_ids
        avg_dwell = np.mean(self.dwell_durations) if self.dwell_durations else 0
        
        return {
            "in": int(self.line_zone.in_count),
            "out": int(self.line_zone.out_count),
            "avg_dwell": float(avg_dwell),
            "active_count": len(current_track_ids)
        }
