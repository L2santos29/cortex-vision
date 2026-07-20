# Performance Profile — Cortex-Vision

## Baseline Benchmarks (2026-07-19)

These benchmarks run automatically with `pytest-benchmark` in `tests/performance/`.
They establish a baseline for detecting regressions.

| Function | Input | Median | P95 | Notes |
|----------|-------|--------|-----|-------|
| `validate_image_content` | JPEG (1KB) | 0.6 µs | 1.0 µs | Magic-byte scan, O(1) |
| `validate_image_content` | PNG (1KB) | 0.8 µs | 1.3 µs | Magic-byte scan |
| `validate_image` | .jpg + 1KB | 5.1 µs | 7.5 µs | Extension + size check |
| `sanitize_filename` | "normal.jpg" | 6.1 µs | 9.2 µs | Regex + truncation |
| `sanitize_filename` | "../special!@#.txt" | 8.1 µs | 12 µs | More replacements |
| `sanitize_filename` | 200 chars | 8.0 µs | 13 µs | Truncation to 128 |
| `frame_to_base64` | 100×100 RGB | 74 µs | 85 µs | JPEG encode + base64 |
| `frame_to_base64` | 640×480→320px | 535 µs | 610 µs | Downscale + encode |
| `draw_boxes` | 0 detections | 0.5 µs | 1.2 µs | Early return |
| `draw_boxes` | 10 detections | 316 µs | 370 µs | Box drawing + labels |

**Total suite**: 10 benchmarks in ~9.5s.

---

## Critical Paths — Known Bottlenecks

### 1. YOLO Inference (`detector.py:Detector.detect`)
- **No automatic benchmark** because it requires the actual model (ultralytics + PyTorch)
- **Estimated**: 50–200ms per image (CPU, YOLOv8n)
- **Variable**: depends on image size, model (n/s/m/l/x), CPU vs GPU
- **How to profile**:
  ```bash
  python -c "
  from src.detector import Detector
  from src.profiling import timed_context
  d = Detector()
  with timed_context('yolo_inference', warn_threshold=1.0):
      r = d.detect('test.jpg')
  print(f'Detections: {len(r)}')
  "
  ```

### 2. Video Processing (`utils.py:process_video_frames`)
- Per frame: extraction → YOLO inference → drawing → base64
- **Dominant**: YOLO inference (50-200ms/frame)
- `frame_to_base64` at 640×480: ~535 µs (negligible vs YOLO)
- `draw_boxes` with 10 detections: ~316 µs (negligible)
- **Scaling**: linear with frame count (1 FPS by default)

### 3. Batch Processing (`pipeline.py:BatchPipeline.process`)
- Sequential iteration over N images
- **Dominant**: N × YOLO inference
- No current parallelization — could benefit from `asyncio.gather` or `ThreadPoolExecutor`

### 4. HTTP Endpoints — Estimated Latency
| Endpoint | No load | Under load | Bottleneck |
|----------|---------|------------|------------|
| `GET /health` | < 5ms | < 10ms | None |
| `GET /` | < 5ms | < 10ms | Static file read |
| `POST /v1/upload` | 100-500ms | 500ms-2s | YOLO inference (dominant) |
| `POST /v1/upload/batch` | N × 100-500ms | variable | Sequential pipeline |
| `POST /v1/upload/video` | F × 100-500ms | variable | YOLO per frame |
| `POST /v1/detect/frame` | 100-500ms | 500ms-2s | YOLO + decode |
| `GET /v1/results/{id}` | < 10ms | < 20ms | Dictionary lookup |

---

## System Profiling

### Using the `@timed` Decorator
```python
@timed(warn_threshold=5.0)
def detect(self, image_path: str) -> list[dict]:
    ...
```
The decorator automatically logs to `logging.INFO` (or `WARNING` if it exceeds the threshold):
```
Profile [Detector.detect]: 0.3452s
Profile [BatchPipeline.process]: 2.1341s
```

It is applied on:
- `detector.py`: `Detector.detect`, `Detector.detect_array`
- `pipeline.py`: `BatchPipeline.process`
- `utils.py`: `process_video_frames`

### Using `timed_context` for Ad-hoc Profiling
```python
from src.profiling import timed_context

with timed_context("custom_operation"):
    # code to measure
    pass
```

### Per-Endpoint Latency Logging
The `MetricsMiddleware` collects per-route latency and exposes it at `/metrics`:
```
cortex_vision_request_latency_seconds{route="/v1/upload",quantile="avg"} 0.3452
cortex_vision_request_latency_seconds{route="/v1/upload",quantile="p99"} 1.2345
```

---

## Performance Improvement Recommendations

1. **Parallelize batch processing**: Use `asyncio.gather` or `ThreadPoolExecutor` to process images in parallel
2. **Cache more aggressively**: `DetectionCache` covers repeated detections, but there's no cache for video frames
3. **Compress/resize before YOLO**: Reduce input resolution to speed up inference
4. **Video streaming**: For long videos, consider async processing with WebSockets
5. **GPU acceleration**: YOLOv8 with CUDA reduces inference from 50-200ms to 5-20ms

---

## How to Run Benchmarks

```bash
# Full benchmarks
API_KEY=test-key pytest tests/performance/ --benchmark-only --benchmark-json=.benchmarks/latest.json

# Compare with baseline (detects regressions)
API_KEY=test-key pytest tests/performance/ --benchmark-compare=.benchmarks/latest.json

# Manual endpoint profiling
curl -w "\n⏱️ Time: %{time_total}s\n" -X POST \
  -H "X-API-Key: $API_KEY" \
  -F "file=@test.jpg" \
  http://localhost:8000/v1/upload
```
