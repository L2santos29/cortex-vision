"""Batch image processing pipeline.

Handles batched detection across multiple images,
result aggregation, and optional LLM-powered scene descriptions.
"""

from .detector import Detector


class BatchPipeline:
    """Process images in batch through detection and aggregation stages.

    Orchestrates batch detection across multiple images, flattens results,
    and computes summary statistics for downstream consumption.
    """

    def __init__(self, detector: Detector) -> None:
        """Prepare the pipeline with a shared detector instance.

        The pipeline coordinates detection across multiple images so callers
        don't manage per-file result merging. The detector is reused for all
        images in the batch.
        """
        self.detector = detector

    def process(self, image_paths: list[str]) -> list[dict]:
        """Run detection on every image path and collect per-file results.

        This method iterates sequentially and can be safely offloaded to a
        thread pool via asyncio.to_thread() since it performs no async I/O.
        """
        results = []
        for path in image_paths:
            detections = self.detector.detect(path)
            results.append({
                "image": path,
                "detections": detections,
                "object_count": len(detections),
            })
        return results

    def aggregate(self, per_file_results: list[dict]) -> list[dict]:
        """Flatten per-file results into a single detection list sorted by confidence.

        Each detection dict is enriched with the source image path so consumers
        can trace results back to origin files.
        """
        aggregated = []
        for file_result in per_file_results:
            for det in file_result["detections"]:
                aggregated.append({
                    "image": file_result["image"],
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                })

        aggregated.sort(key=lambda d: d["confidence"], reverse=True)
        return aggregated

    def stats(self, aggregated: list[dict]) -> dict:
        """Compute per-class counts and summary statistics.

        Returns total detections, unique class count, a per-class histogram,
        and the top-5 most frequent classes — useful for the batch results
        dashboard and CSV export.
        """
        class_counts = {}
        total = len(aggregated)

        for det in aggregated:
            cls = det["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        return {
            "total_detections": total,
            "unique_classes": len(class_counts),
            "per_class": class_counts,
            "top_classes": sorted(
                class_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }
