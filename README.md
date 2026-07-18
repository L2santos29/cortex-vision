# 🧠 Cortex-Vision

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Object Recognition Platform** — Upload images or videos, or use your webcam. YOLOv8 detects objects in real time with bounding boxes, confidence scores, and batch processing.

> Full-stack: FastAPI backend + vanilla JS frontend + YOLOv8 inference.

---

## Features

| Feature | Description |
|---------|-------------|
| **📸 Batch Image Analysis** | Upload single or multiple images (up to 500+). Each image is processed with bounding boxes drawn directly on the preview. |
| **🎬 Video Analysis** | Upload MP4, AVI, MOV, MKV, or WebM. Video is sampled at 1 FPS, each frame is analyzed with bounding boxes overlaid. Interactive timeline for quick navigation. |
| **📹 Live Webcam** | Start your camera and see detections update every second. Bounding boxes are drawn live over the video feed. |
| **📊 Class Distribution Chart** | Collapsible bar chart showing detected class frequency with gradient bars (red → yellow → green). |
| **🎚️ Confidence Filter** | Slider to dynamically hide/show detections below a confidence threshold. |
| **📥 Export** | Download results as CSV or download individual annotated images with bounding boxes. |
| **📋 Session History** | Past detection sessions are saved in your browser (localStorage). Browse and revisit them anytime. |
| **🌙☀️ Theme Toggle** | Switch between dark and light themes. Preference is persisted across sessions. |
| **⌨️ Keyboard Shortcuts** | `Space` (webcam), `E` (export), `R` (clear), `1`-`5` (tab navigation). |
| **🎯 Sidebar Navigation** | Tab-based layout with isolated sections for Images, Video, Webcam, History, and About. |

---

## Stack

| Layer | Tech |
|-------|------|
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Backend** | FastAPI + Python 3.12+ |
| **Frontend** | Vanilla HTML/CSS/JS (no framework) — canvas-based rendering with bounding boxes |
| **Image/Video Processing** | OpenCV, NumPy |
| **Deployment** | Docker |

---

## Quick Start

### Using Make (recommended)

```bash
git clone https://github.com/L2santos29/cortex-vision.git
cd cortex-vision
make run
```

The Makefile automatically creates a virtual environment and installs dependencies. Open **http://localhost:8000**.

### Manual

```bash
# Clone
git clone https://github.com/L2santos29/cortex-vision.git
cd cortex-vision

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU-only, no CUDA required)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Run
python -m src.main
```

### Docker

```bash
docker build -t cortex-vision .
docker run -p 8000:8000 cortex-vision
```

---

## Project Structure

```
cortex-vision/
├── src/
│   ├── main.py          # FastAPI app — routes, webcam endpoint
│   ├── detector.py       # YOLOv8 inference engine (detect + detect_array)
│   ├── pipeline.py       # Batch processing and aggregation
│   └── utils.py          # Validation, image preprocessing, video frame extraction
├── static/
│   └── index.html        # Full web UI — tabs, canvas, webcam, history
├── tests/
│   └── test_detector.py
├── scripts/
│   └── run.sh            # Quick start helper
├── Makefile              # Entry point: make run, make test, make clean
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/upload` | Upload single image |
| `POST` | `/v1/upload/batch` | Upload multiple images (batch) |
| `POST` | `/v1/upload/video` | Upload video — returns per-frame detections with annotated frame images |
| `POST` | `/v1/detect/frame` | Receive a webcam frame and run detection (no disk I/O) |
| `GET` | `/v1/results/{task_id}` | Get batch image results |
| `GET` | `/v1/export/{task_id}` | Export batch results as CSV |
| `GET` | `/health` | Health check |

---

## Quick API Tests

```bash
# Health
curl http://localhost:8000/health

# Upload image
curl -X POST -H "X-API-Key: your-api-key" -F "file=@image.jpg" http://localhost:8000/v1/upload

# Batch upload
curl -X POST -H "X-API-Key: your-api-key" -F "files=@img1.jpg" -F "files=@img2.jpg" http://localhost:8000/v1/upload/batch

# Video upload
curl -X POST -H "X-API-Key: your-api-key" -F "file=@video.mp4" http://localhost:8000/v1/upload/video

# Webcam frame
curl -X POST -H "X-API-Key: your-api-key" -F "file=@frame.jpg" http://localhost:8000/v1/detect/frame

# Export CSV
curl http://localhost:8000/v1/export/<task_id> -o results.csv
```

---

## Requirements

- Python 3.12+
- 4GB+ RAM recommended (YOLOv8 inference)
- No GPU required (CPU-only PyTorch, works on any machine)
- Webcam required for live detection feature
- Tested on Linux. Windows/macOS should work with minor adjustments.

---

## Status

**Pre-Alpha** — Core detection pipeline is functional with all three input modes (images, video, webcam). UI is fully interactive with bounding boxes, filters, timeline, and history. Areas for improvement: authentication, rate limiting, persistent storage, and comprehensive test coverage.

---

## Configuration

The following environment variables can be used to configure the application:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | _(required)_ | API key for authentication on protected endpoints |
| `CORS_ORIGINS` | `http://localhost:8000` | Comma-separated list of allowed CORS origins |
| `RATE_LIMIT` | `30` | Maximum requests per IP per rate window |
| `RATE_WINDOW` | `60` | Rate limit window in seconds |
| `UPLOAD_DIR` | `uploads` | Directory for uploaded files |
| `OUTPUT_DIR` | `output` | Directory for output files |
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model variant |

## License

MIT — See [LICENSE](LICENSE)
