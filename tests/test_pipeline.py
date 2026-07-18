"""Tests for the batch processing pipeline."""

from unittest.mock import MagicMock

import pytest

from src.pipeline import BatchPipeline


def test_batch_pipeline_init():
    """Pipeline stores the detector reference."""
    detector = MagicMock()
    pipeline = BatchPipeline(detector)
    assert pipeline.detector is detector


def test_batch_pipeline_process(mock_yolo, monkeypatch):
    """process runs detection on each image and returns correct structure."""
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    from tests.conftest import MockResults

    mock_yolo.side_effect = [
        [
            MockResults(
                names={0: "person"},
                boxes_data=[(0, 0.95, [10.0, 20.0, 100.0, 200.0])],
            )
        ],
        [
            MockResults(
                names={2: "car"},
                boxes_data=[(2, 0.85, [50.0, 60.0, 300.0, 150.0])],
            )
        ],
    ]

    from src.detector import Detector

    real_detector = Detector()
    pipeline = BatchPipeline(real_detector)
    results = pipeline.process(["img1.jpg", "img2.jpg"])

    assert len(results) == 2
    assert results[0]["image"] == "img1.jpg"
    assert results[0]["object_count"] == 1
    assert results[0]["detections"][0]["class"] == "person"

    assert results[1]["image"] == "img2.jpg"
    assert results[1]["object_count"] == 1
    assert results[1]["detections"][0]["class"] == "car"


def test_batch_pipeline_aggregate(sample_detections):
    """aggregate flattens per-file results into a single list."""
    detector = MagicMock()
    pipeline = BatchPipeline(detector)

    per_file_results = [
        {
            "image": "img1.jpg",
            "detections": sample_detections[:2],
            "object_count": 2,
        },
        {
            "image": "img2.jpg",
            "detections": sample_detections[2:],
            "object_count": 1,
        },
    ]

    aggregated = pipeline.aggregate(per_file_results)

    assert len(aggregated) == 3
    # The aggregation process adds 'image' key to each detection
    assert aggregated[0]["image"] == "img1.jpg"
    assert aggregated[0]["class"] == "person"
    assert aggregated[1]["image"] == "img1.jpg"
    assert aggregated[1]["class"] == "car"
    assert aggregated[2]["image"] == "img2.jpg"
    assert aggregated[2]["class"] == "dog"


def test_batch_pipeline_aggregate_sorted():
    """aggregate sorts results by confidence descending."""
    detector = MagicMock()
    pipeline = BatchPipeline(detector)

    per_file_results = [
        {
            "image": "test.jpg",
            "detections": [
                {"class": "dog", "confidence": 0.72, "bbox": [0, 0, 10, 10]},
                {"class": "car", "confidence": 0.85, "bbox": [0, 0, 10, 10]},
                {"class": "person", "confidence": 0.95, "bbox": [0, 0, 10, 10]},
            ],
            "object_count": 3,
        },
    ]

    aggregated = pipeline.aggregate(per_file_results)

    assert len(aggregated) == 3
    assert aggregated[0]["class"] == "person"
    assert aggregated[1]["class"] == "car"
    assert aggregated[2]["class"] == "dog"


def test_batch_pipeline_stats():
    """stats computes class counts and summary correctly."""
    detector = MagicMock()
    pipeline = BatchPipeline(detector)

    aggregated = [
        {"image": "img1.jpg", "class": "person", "confidence": 0.95, "bbox": [0, 0, 10, 10]},
        {"image": "img1.jpg", "class": "person", "confidence": 0.90, "bbox": [0, 0, 10, 10]},
        {"image": "img2.jpg", "class": "car", "confidence": 0.85, "bbox": [0, 0, 10, 10]},
        {"image": "img2.jpg", "class": "dog", "confidence": 0.72, "bbox": [0, 0, 10, 10]},
    ]

    stats = pipeline.stats(aggregated)

    assert stats["total_detections"] == 4
    assert stats["unique_classes"] == 3
    assert stats["per_class"] == {"person": 2, "car": 1, "dog": 1}
    assert len(stats["top_classes"]) == 3
    assert stats["top_classes"][0] == ("person", 2)


def test_batch_pipeline_process_empty_list(mock_yolo):
    """process handles empty list of paths gracefully."""
    from src.detector import Detector

    pipeline = BatchPipeline(Detector())
    results = pipeline.process([])
    assert results == []


def test_batch_pipeline_aggregate_empty():
    """aggregate handles empty per_file_results."""
    detector = MagicMock()
    pipeline = BatchPipeline(detector)
    aggregated = pipeline.aggregate([])
    assert aggregated == []


def test_batch_pipeline_stats_empty():
    """stats handles empty aggregated list."""
    detector = MagicMock()
    pipeline = BatchPipeline(detector)
    stats = pipeline.stats([])
    assert stats["total_detections"] == 0
    assert stats["unique_classes"] == 0
    assert stats["per_class"] == {}
    assert stats["top_classes"] == []


def test_batch_pipeline_stats_single_class():
    """stats with all same class returns correct counts."""
    detector = MagicMock()
    pipeline = BatchPipeline(detector)
    aggregated = [
        {"image": "img1.jpg", "class": "person", "confidence": 0.95, "bbox": [0, 0, 10, 10]},
        {"image": "img1.jpg", "class": "person", "confidence": 0.90, "bbox": [0, 0, 10, 10]},
    ]
    stats = pipeline.stats(aggregated)
    assert stats["unique_classes"] == 1
    assert stats["per_class"] == {"person": 2}
