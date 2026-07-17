"""Cortex-Vision main entry point — FastAPI application."""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .detector import Detector
from .pipeline import BatchPipeline

app = FastAPI(title="Cortex-Vision", version="0.1.0")

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Engine
detector = Detector()
pipeline = BatchPipeline(detector)

# Store batch results in memory (in production: Redis/DB)
batch_results = {}

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<h1>Cortex-Vision API</h1><p>Static UI not found. Use /docs for API.</p>"


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload a single image and get detection results."""
    contents = await file.read()
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    image_path.write_bytes(contents)

    results = detector.detect(str(image_path))

    return JSONResponse({
        "filename": file.filename,
        "detections": results,
    })


@app.post("/upload/batch")
async def upload_batch(files: list[UploadFile]):
    """Upload multiple images for batch processing."""
    task_id = uuid.uuid4().hex
    file_paths = []

    for file in files:
        contents = await file.read()
        image_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
        image_path.write_bytes(contents)
        file_paths.append(str(image_path))

    # Process batch asynchronously
    results = pipeline.process(file_paths)
    aggregated = pipeline.aggregate(results)

    batch_results[task_id] = {
        "files_processed": len(files),
        "detections": aggregated,
        "per_file": results,
    }

    return JSONResponse({
        "task_id": task_id,
        "files_processed": len(files),
        "total_objects": len(aggregated),
    })


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """Get batch processing results."""
    if task_id not in batch_results:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(batch_results[task_id])


@app.get("/export/{task_id}")
async def export_csv(task_id: str):
    """Export batch results as CSV."""
    if task_id not in batch_results:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    import csv
    import io

    detections = batch_results[task_id]["detections"]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["image", "class", "confidence", "bbox"])

    for d in detections:
        writer.writerow([
            d.get("image", ""),
            d.get("class", ""),
            d.get("confidence", 0),
            d.get("bbox", []),
        ])

    csv_path = OUTPUT_DIR / f"{task_id}.csv"
    csv_path.write_text(output.getvalue())

    return FileResponse(csv_path, filename=f"cortex-vision-{task_id}.csv")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": detector.model_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
