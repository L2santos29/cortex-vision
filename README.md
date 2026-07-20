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
│   ├── main.py           # FastAPI app — routes, middleware setup
│   ├── config.py         # Pydantic settings from env vars
│   ├── detector.py       # YOLOv8 inference engine (detect + detect_array)
│   ├── pipeline.py       # Batch processing and aggregation
│   ├── services.py       # Service layer — upload, validation, detection orchestration
│   ├── cache.py          # LRU detection cache to avoid redundant inference
│   ├── resilience.py     # Circuit breaker + retry with exponential backoff
│   ├── middleware.py     # Auth, CORS, rate limiting, security headers, metrics, tracing
│   ├── profiling.py      # Performance decorators (@timed) and context managers
│   ├── utils.py          # Validation, magic-byte checks, frame extraction, drawing
│   └── cli.py            # Command-line interface (detect, serve subcommands)
├── static/
│   └── index.html        # Full web UI — tabs, canvas, webcam, history
├── tests/
│   ├── smoke/            # Module imports, app instantiation
│   ├── unit/             # Unit tests for services, routes, profiling, utils
│   ├── edge/             # Edge cases — nulls, boundaries, malformed inputs
│   ├── integration/      # API + middleware integration tests
│   ├── contract/         # API response schema and status code checks
│   ├── property/         # Hypothesis property-based tests
│   ├── regression/       # Regression tests for known bugs
│   ├── performance/      # pytest-benchmark baselines
│   └── fault/            # Fault injection — disk full, network timeout, bad auth
├── deploy/
│   ├── docker-compose.staging.yml  # Staging environment with Caddy + TLS
│   ├── Caddyfile         # Reverse proxy config with Let's Encrypt
│   ├── prometheus-alerts.yml       # Alerting rules for production
│   └── grafana-dashboard.json      # Grafana dashboard definition
├── .github/
│   ├── workflows/ci.yml            # CI — tests + coverage + lint + dependency audit
│   ├── workflows/deploy-staging.yml # Staging deployment pipeline
│   └── dependabot.yml              # Automated dependency updates
├── scripts/
│   └── run.sh            # Quick start helper
├── PERFORMANCE.md         # Benchmark baselines and bottleneck analysis
├── Makefile               # Entry point: make run, make test, make clean
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
| `GET` | `/v1/results/{task_id}` | Get batch image results (paginated) |
| `GET` | `/v1/export/{task_id}` | Export batch results as CSV |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics (request count, latency, P99) |

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
curl -H "X-API-Key: your-api-key" http://localhost:8000/v1/export/<task_id> -o results.csv

# Metrics (Prometheus format)
curl -H "X-API-Key: your-api-key" http://localhost:8000/metrics
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

**Alpha** — Core detection pipeline is functional with all three input modes (images, video, webcam). Authentication, rate limiting, security headers, metrics, and alerting are all implemented. Test coverage is at **87%** with 290 tests across unit, integration, contract, property-based, regression, and fault injection suites.

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

### Environment File (`.env.example`)

A `.env.example` file is included in the repository. Copy it to `.env` and adjust values:

```bash
cp .env.example .env
```

The application reads environment variables from `.env` automatically via `pydantic-settings`. All values in `.env.example` are safe to commit — it contains only placeholder keys for local development.

> **Security note:** Never commit the actual `.env` file. Add it to `.gitignore` if it isn't already.

---

## Staging Environment

Usa la configuración específica de staging con TLS automático, límites de recursos y health checks:

```bash
# Requisitos: API_KEY definida
export API_KEY=$(openssl rand -hex 32)

# Iniciar entorno staging (Caddy + App)
docker compose -f deploy/docker-compose.staging.yml up -d

# Verificar estado
curl -f http://localhost:8000/health

# Ver logs
docker compose -f deploy/docker-compose.staging.yml logs -f
```

El compose de staging incluye:
- **Caddy** como reverse proxy con TLS automático (Let's Encrypt)
- **Límites de memoria** (2GB max, 512MB reservados)
- **Límites de CPU** (2 cores max, 0.5 reservados)
- **Política de restart** (`unless-stopped`)
- **Health check** cada 30s para orquestación
- **Logging** con rotación (10MB por archivo, max 3 archivos)
- **Red aislada** (bridge)

### Despliegue automatizado (CI/CD)

El workflow `.github/workflows/deploy-staging.yml` se ejecuta automáticamente
en cada push a `main`:
1. Construye la imagen Docker
2. Guarda el artefacto para deploy
3. (Opcional) Despliega vía SSH al servidor de staging

> Los tests con cobertura se ejecutan en `.github/workflows/ci.yml` y son requisito
> para que el deploy proceda.

Para activar el deploy remoto, configura los secrets en GitHub:
- `STAGING_HOST`
- `STAGING_USER`
- `STAGING_SSH_KEY`

### Staging Checklist

- [ ] Generar `API_KEY` fuerte: `openssl rand -hex 32`
- [ ] Configurar `CORS_ORIGINS` con el dominio de staging
- [ ] Editar `deploy/Caddyfile` con el dominio real
- [ ] Verificar que el health check responde
- [ ] Monitorear `/health` y `/metrics`

---

## Rollback Approach

### Code Rollback

```bash
# Roll back to a specific tag or commit
git checkout v0.1.0
docker build -t cortex-vision:v0.1.0 .
docker stop cortex-vision-staging
docker run -d \
  --name cortex-vision-staging \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  cortex-vision:v0.1.0
```

### Docker Image Rollback

If using a container registry with tagged images:

```bash
docker pull myregistry/cortex-vision:v0.1.0
docker stop cortex-vision-staging
docker run -d --name cortex-vision-staging --restart unless-stopped -p 8000:8000 --env-file .env myregistry/cortex-vision:v0.1.0
```

### Rollback Considerations

- **Database:** The service is stateless with respect to detections — no database migrations to revert.
- **Uploaded files:** Rollback does not affect previously uploaded files stored in `UPLOAD_DIR` / `OUTPUT_DIR`.
- **Client compatibility:** API changes are versioned under `/v1/`. A rollback preserves backward compatibility for existing clients.
- **Health check:** Always verify `/health` after rollback before directing traffic.

---

## License

MIT — See [LICENSE](LICENSE)
