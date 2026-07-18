"""Lightweight profiling utilities for measuring endpoint and pipeline performance.

Usage:
    @timed(level=logging.INFO)
    def my_function():
        ...

    with timed_context("detection") as ctx:
        result = detect(image)
"""

import asyncio
import functools
import logging
import time

logger = logging.getLogger(__name__)


class timed_context:
    """Context manager that logs execution time of a code block.

    Args:
        name: Label for the operation being timed.
        level: Log level (default DEBUG).
        warn_threshold: If set, log at WARNING when elapsed exceeds this (seconds).

    Example:
        with timed_context("yolo_inference"):
            results = model(image)
    """

    def __init__(self, name: str, level: int = logging.DEBUG, warn_threshold: float | None = None):
        self.name = name
        self.level = level
        self.warn_threshold = warn_threshold
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.warn_threshold and self.elapsed > self.warn_threshold:
            logger.warning("Profile [%s]: %.4fs", self.name, self.elapsed)
        else:
            logger.log(self.level, "Profile [%s]: %.4fs", self.name, self.elapsed)


def timed(name: str | None = None, level: int = logging.DEBUG, warn_threshold: float | None = None):
    """Decorator that logs execution time of the wrapped function.

    Args:
        name: Label for the operation (defaults to function name).
        level: Log level.
        warn_threshold: Log at WARNING when elapsed exceeds this (seconds).

    Example:
        @timed(warn_threshold=1.0)
        def detect(image):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            label = name or func.__qualname__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log_fn = logger.warning if (warn_threshold and elapsed > warn_threshold) else logger.log
            if log_fn is logger.warning:
                log_fn("Profile [%s]: %.4fs", label, elapsed)
            else:
                log_fn(level, "Profile [%s]: %.4fs", label, elapsed)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            label = name or func.__qualname__
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log_fn = logger.warning if (warn_threshold and elapsed > warn_threshold) else logger.log
            if log_fn is logger.warning:
                log_fn("Profile [%s]: %.4fs", label, elapsed)
            else:
                log_fn(level, "Profile [%s]: %.4fs", label, elapsed)
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator
