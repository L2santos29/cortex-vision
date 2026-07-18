# Performance Profile — Cortex-Vision

## Baseline Benchmarks (2026-07-19)

Estos benchmarks se ejecutan automáticamente con `pytest-benchmark` en `tests/performance/`.
Se establecen como línea base para detectar regresiones.

| Función | Input | Mediana | P95 | Observaciones |
|---------|-------|---------|-----|---------------|
| `validate_image_content` | JPEG (1KB) | 0.6 µs | 1.0 µs | Magic-byte scan, O(1) |
| `validate_image_content` | PNG (1KB) | 0.8 µs | 1.3 µs | Magic-byte scan |
| `validate_image` | .jpg + 1KB | 5.1 µs | 7.5 µs | Extension + size check |
| `sanitize_filename` | "normal.jpg" | 6.1 µs | 9.2 µs | Regex + truncation |
| `sanitize_filename` | "../special!@#.txt" | 8.1 µs | 12 µs | More replacements |
| `sanitize_filename` | 200 chars | 8.0 µs | 13 µs | Truncation to 128 |
| `frame_to_base64` | 100×100 RGB | 74 µs | 85 µs | JPEG encode + base64 |
| `frame_to_base64` | 640×480→320px | 535 µs | 610 µs | Downscale + encode |
| `draw_boxes` | 0 detecciones | 0.5 µs | 1.2 µs | Early return |
| `draw_boxes` | 10 detecciones | 316 µs | 370 µs | Box drawing + labels |

**Total suite**: 10 benchmarks en ~9.5s.

---

## Critical Paths — Known Bottlenecks

### 1. YOLO Inference (`detector.py:Detector.detect`)
- **Sin benchmark automático** porque requiere el modelo real (ultralytics + PyTorch)
- **Estimado**: 50–200ms por imagen (CPU, YOLOv8n)
- **Variable**: depende del tamaño de imagen, modelo (n/s/m/l/x), CPU vs GPU
- **Cómo perfilar**:
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
- Por cada frame: extracción → YOLO inference → dibujo → base64
- **Dominante**: YOLO inference (50-200ms/frame)
- `frame_to_base64` en 640×480: ~535 µs (despreciable contra YOLO)
- `draw_boxes` con 10 detecciones: ~316 µs (despreciable)
- **Escalado**: lineal con número de frames (1 FPS por defecto)

### 3. Batch Processing (`pipeline.py:BatchPipeline.process`)
- Iteración secuencial sobre N imágenes
- **Dominante**: N × YOLO inference
- Sin paralelización actual — podría beneficiarse de `asyncio.gather` o `ThreadPoolExecutor`

### 4. Endpoints HTTP — Latencia estimada
| Endpoint | Sin carga | Bajo carga | Cuello de botella |
|----------|-----------|------------|-------------------|
| `GET /health` | < 5ms | < 10ms | Ninguno |
| `GET /` | < 5ms | < 10ms | Lectura de archivo estático |
| `POST /v1/upload` | 100-500ms | 500ms-2s | YOLO inference (dominante) |
| `POST /v1/upload/batch` | N × 100-500ms | variable | Pipeline secuencial |
| `POST /v1/upload/video` | F × 100-500ms | variable | YOLO por frame |
| `POST /v1/detect/frame` | 100-500ms | 500ms-2s | YOLO + decode |
| `GET /v1/results/{id}` | < 10ms | < 20ms | Búsqueda en diccionario |

---

## Profiling del Sistema

### Uso del decorador `@timed`
```python
@timed(warn_threshold=5.0)
def detect(self, image_path: str) -> list[dict]:
    ...
```
El decorador registra automáticamente en `logging.INFO` (o `WARNING` si excede el umbral):
```
Profile [Detector.detect]: 0.3452s
Profile [BatchPipeline.process]: 2.1341s
```

Está aplicado en:
- `detector.py`: `Detector.detect`, `Detector.detect_array`
- `pipeline.py`: `BatchPipeline.process`
- `utils.py`: `process_video_frames`

### Uso de `timed_context` para profiling ad-hoc
```python
from src.profiling import timed_context

with timed_context("custom_operation"):
    # código a medir
    pass
```

### Logging de latencia por endpoint
El `MetricsMiddleware` recolecta latencia por ruta y la expone en `/metrics`:
```
cortex_vision_request_latency_seconds{route="/v1/upload",quantile="avg"} 0.3452
cortex_vision_request_latency_seconds{route="/v1/upload",quantile="p99"} 1.2345
```

---

## Recomendaciones para Mejora de Performance

1. **Paralelizar batch processing**: Usar `asyncio.gather` o `ThreadPoolExecutor` para procesar imágenes en paralelo
2. **Cachear más agresivamente**: `DetectionCache` cubre detecciones repetidas, pero no hay cache para video frames
3. **Comprimir/resize antes de YOLO**: Reducir resolución de entrada para acelerar inference
4. **Streaming de video**: Para videos largos, considerar procesamiento asíncrono con WebSockets
5. **GPU acceleration**: YOLOv8 con CUDA reduce inference de 50-200ms a 5-20ms

---

## Cómo ejecutar los benchmarks

```bash
# Benchmarks completos
API_KEY=test-key pytest tests/performance/ --benchmark-only --benchmark-json=.benchmarks/latest.json

# Comparar con línea base (detecta regresiones)
API_KEY=test-key pytest tests/performance/ --benchmark-compare=.benchmarks/latest.json

# Profiling manual de un endpoint
curl -w "\n⏱️ Time: %{time_total}s\n" -X POST \
  -H "X-API-Key: $API_KEY" \
  -F "file=@test.jpg" \
  http://localhost:8000/v1/upload
```
