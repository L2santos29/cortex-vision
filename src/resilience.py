"""Resilience utilities: retry with backoff and circuit breaker."""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    **kwargs: Any,
) -> Any:
    """Retry an async function with exponential backoff and jitter.

    Args:
        func: Async callable to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except (OSError, ConnectionError, TimeoutError) as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                # Jitter prevents thundering herd when multiple clients retry simultaneously
                jitter = delay * 0.25 * (2 * random.random() - 1)
                await asyncio.sleep(delay + jitter)
            else:
                raise last_exc


class CircuitBreaker:
    """Simple circuit breaker to prevent cascading failures.

    Tracks consecutive failures and opens the circuit when a threshold
    is reached, allowing the system to recover before retrying.
    """

    OPEN = "open"
    HALF_OPEN = "half-open"
    CLOSED = "closed"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = self.CLOSED
        self.last_failure_time = 0.0

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute func if circuit is closed, raise RuntimeError if open."""
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = self.HALF_OPEN
            else:
                raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self.failure_count = 0
            return result
        except Exception as exc:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
            raise exc

    async def async_call(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """Async version of call()."""
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = self.HALF_OPEN
            else:
                raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self.failure_count = 0
            return result
        except Exception as exc:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
            raise exc
