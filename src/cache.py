import hashlib
from collections import OrderedDict

import numpy as np

CACHE_HASH_SAMPLE_BYTES = 4096  # First 4 KiB of array content for cache key


class DetectionCache:
    """Simple LRU cache for detection results to avoid redundant inference."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()

    def _make_key(self, image_path: str) -> str:
        return f"path:{image_path}"

    def _make_array_key(self, img: np.ndarray) -> str:
        # Sample first 4 KiB of content — avoids hashing the entire array
        return f"arr:{hashlib.md5(img.tobytes()[:CACHE_HASH_SAMPLE_BYTES]).hexdigest()}"

    def get(self, key: str) -> list | None:
        """Get cached result and mark as recently used."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: list) -> None:
        """Store result in cache, evicting oldest if at capacity."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
