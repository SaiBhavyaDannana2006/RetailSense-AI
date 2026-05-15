# RetailSense AI: Real-Time Video Intelligence & Analytics Engine

RetailSense AI is a high-performance computer vision system designed to transform raw retail video feeds into actionable business intelligence. By integrating real-time Multi-Object Tracking (MOT) with Large Language Model (LLM) reasoning, the system identifies behavioral patterns and "Lost Conversion" opportunities in physical retail spaces.

## 🏗️ System Architecture
The project follows a decoupled, production-ready architecture:

- **Edge Inference:** Optimized YOLOv8 model running on ONNX Runtime for low-latency person detection and tracking.
- **Backend API:** Asynchronous FastAPI server handling WebSocket streams and data persistence.
- **Intelligence Layer:** Google Gemini 3.0 integration for contextual analysis of customer dwell times.
- **Data Persistence:** SQLite engine for historical logging and analytics retrieval.
- **Frontend:** Real-time dashboard with live telemetry and automated insights.



## 🚀 Key Features
- **Low-Latency Tracking:** Sub-100ms inference using quantized ONNX weights.
- **Persistent Analytics:** Historical tracking of entries, exits, and average dwell times.
- **Automated Insights:** AI-generated conversion reports based on customer behavior.
- **Containerized:** Fully reproducible environment using Docker and Docker Compose.

## 🛠️ Tech Stack
- **Languages:** Python 3.11+, JavaScript
- **ML Frameworks:** Ultralytics (YOLOv8), ONNX Runtime, ByteTrack
- **Backend:** FastAPI, Uvicorn, SQLAlchemy
- **Generative AI:** Google GenAI (Gemini 3.0)
- **DevOps:** Docker, GitHub Actions

## 📦 Installation & Setup

### Prerequisites
- Docker & Docker Compose
- Google Gemini API Key

### Deployment
1. Clone the repository:
   ```bash
   git clone [https://github.com/SaiBhavyaDannana2006/RetailSense-AI.git](https://github.com/SaiBhavyaDannana2006/RetailSense-AI.git)
   cd RetailSense-AIS