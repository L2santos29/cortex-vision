# 🧠 Cortex-Vision

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-7C3AED)](https://docs.ultralytics.com/)
[![Status](https://img.shields.io/badge/Status-Pre--Alpha-FF6B35)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Object Recognition SaaS with Batch Processing and LLM-Powered Scene Understanding**

> Upload images → YOLOv8 detects objects → batch pipeline processes hundreds of images → web dashboard shows results. Full-stack: FastAPI backend + web frontend + LLM integration.

---

## 🎯 What is Cortex-Vision?

Cortex-Vision is an end-to-end object recognition platform built around a core idea: **CV + LLM orchestration as a service.**

- **Upload** one or hundreds of images via web UI or REST API
- **Detect** objects using YOLOv8 (COCO dataset — 80 classes)
- **Process in batch** with a configurable pipeline (extraction, filtering, aggregation)
- **Describe** scenes with LLM integration for richer results
- **Export** results as CSV/JSON reports
- **Deploy** with a single Docker command

---

## 🔧 Stack

| Layer | Tech |
|-------|------|
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Backend** | FastAPI + Python 3.12+ |
| **LLM Integration** | LangChain (optional scene description) |
| **Frontend** | HTML/CSS/JS (vanilla, no framework) |
| **Data Processing** | NumPy, OpenCV, Pillow |
| **Deployment** | Docker, Docker Compose |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/L2santos29/cortex-vision.git
cd cortex-vision

# Install
pip install -r requirements.txt

# Run
python -m src.main

# Open http://localhost:8000
```

### Docker

```bash
docker build -t cortex-vision .
docker run -p 8000:8000 cortex-vision
```

---

## 📁 Project Structure

```
cortex-vision/
├── src/
│   ├── main.py          # FastAPI app entry point
│   ├── detector.py       # YOLOv8 inference engine
│   ├── pipeline.py       # Batch processing pipeline
│   └── utils.py          # Image preprocessing, helpers
├── static/
│   └── index.html        # Web UI frontend
├── tests/
│   └── test_detector.py
├── scripts/
│   └── run.sh
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Batch Processing Pipeline

```
User uploads N images
        ↓
[Preprocessing] → Resize, validate, normalize
        ↓
[Detection] → YOLOv8 inference on each image
        ↓
[Aggregation] → Collect results (class, confidence, count, bbox)
        ↓
[LLM Description] → (Optional) Scene-level description per image
        ↓
[Export] → CSV/JSON report + annotated images
```

---

## 🔑 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload single image |
| `POST` | `/upload/batch` | Upload multiple images (batch) |
| `GET` | `/results/{task_id}` | Get results for a batch job |
| `GET` | `/export/{task_id}` | Export results as CSV |
| `GET` | `/health` | Health check |

---

## 📦 Requirements

- Python 3.12+
- 4GB+ RAM recommended (YOLOv8 inference)
- GPU optional (auto-detected if available)

---

## 🏗️ Status

**Pre-Alpha** — Core detection pipeline functional. Currently implementing batch processing UI and LLM integration layer.

---

## 📄 License

MIT — See [LICENSE](LICENSE)
