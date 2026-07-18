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
        """Store a configured Detector instance for batch processing.

        The pipeline coordinates detection, aggregation, and statistics
        so that callers don't need to manage per-file result merging.

        Args:
            detector: Configured Detector instance.
        """
        self.detector = detector

    def process(self, image_paths: list[str]) -> list[dict]:
        """Run detection on every image path and collect per-file results.

        Each result dict carries the image path, the list of detections,
        and a convenience count.  This method is intentionally a simple
        loop so it can be safely offloaded to a thread pool.

        Args:
            image_paths: List of paths to image files.

        Returns:
            List of per-image result dicts.
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
        """Flatten per-file results into a single sorted detection list.

        Each detection is enriched with its source image path so downstream
        consumers can trace results back to origin files.  The list is
        sorted by confidence (highest first) for convenient display.

        Args:
            per_file_results: Output from self.process().

        Returns:
            Flat list of detection dicts with image source.
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
        """Compute per-class counts and summary statistics (used by /v1/upload/batch).

        Aggregated detection lists can be large; this method condenses them
        into a lightweight summary: total detections, unique class count,
        per-class histogram, and the top-5 most frequent classes.

        Args:
            aggregated: Output from self.aggregate().

        Returns:
            Dict with totals, per-class counts, and confidence stats.
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
