"""Batch image processing pipeline.

Handles batched detection across multiple images,
result aggregation, and optional LLM-powered scene descriptions.
"""

from .detector import Detector


class BatchPipeline:
    """Process images in batch through detection and aggregation stages."""

    def __init__(self, detector: Detector):
        """Initialize pipeline with a detector instance.

        Args:
            detector: Configured Detector instance.
        """
        self.detector = detector

    def process(self, image_paths: list[str]) -> list[dict]:
        """Run detection on a batch of images.

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
        """Aggregate detection results across all images.

        Args:
            per_file_results: Output from self.process().

        Returns:
            Flat list of detections with image source.
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

        # Sort by confidence (highest first)
        aggregated.sort(key=lambda d: d["confidence"], reverse=True)
        return aggregated

    def stats(self, aggregated: list[dict]) -> dict:
        """Compute summary statistics from aggregated results.

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
