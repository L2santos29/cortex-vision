"""Utility functions for image preprocessing and helpers."""

from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_image(filename: str, size: int) -> None:
    """Validate uploaded image file.

    Args:
        filename: Original filename.
        size: File size in bytes.

    Raises:
        HTTPException: If validation fails.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    if size > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large (>10MB)")


def preprocess(image_path: str, target_size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """Load and preprocess an image for inference.

    Args:
        image_path: Path to the image.
        target_size: Desired output size (width, height).

    Returns:
        Preprocessed image as numpy array (BGR).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, target_size)
    return img
