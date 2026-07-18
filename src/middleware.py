"""Middleware, auth dependency, and metrics storage for the FastAPI application."""

import logging
import statistics
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable

from fastapi import Depends, HTTPException, Security
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

# Type alias for the ASGI call_next parameter in middleware
NextCall = Callable[[Request], Awaitable[Response]]

from .config import settings

logger = logging.getLogger(__name__)

# ---- Metrics storage (MON-03) ----
REQUEST_COUNT: dict[str, int] = defaultdict(int)
REQUEST_LATENCY: dict[str, list[float]] = defaultdict(list)


# ---- Auth dependency (SEC-02) ----
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str = Security(api_key_header),
    request: Request = None,
) -> str:
    """Dependency that validates API key for non-public endpoints.

    Checks the ``X-API-Key`` header first, then falls back to the ``api_key``
    HTTP-only cookie (set by the index page for the web UI).
    """
    if api_key == settings.api_key:
        return api_key
    # Fallback: check the HTTP-only cookie (set by the web UI)
    if request is not None:
        cookie_key = request.cookies.get("api_key")
        if cookie_key == settings.api_key:
            return cookie_key
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---- Middleware: HTTPS Redirect (SEC-10) ----
class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect HTTP → HTTPS when behind a TLS-terminating reverse proxy.

    The reverse proxy (nginx, Caddy, Traefik) sets ``X-Forwarded-Proto`` to
    indicate the original protocol. When it is "http", we redirect to HTTPS.
    Direct connections without the header are left untouched (dev mode).
    """

    async def dispatch(self, request: Request, call_next: NextCall):
        forwarded_proto = request.headers.get("x-forwarded-proto", "")
        if forwarded_proto and forwarded_proto != "https":
            url = str(request.url).replace("http://", "https://", 1)
            return RedirectResponse(url, status_code=301)
        return await call_next(request)


# ---- Middleware: Security Headers (SEC-06) ----
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Set security headers on every response."""

    async def dispatch(self, request: Request, call_next: NextCall):
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'"
        )
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


# ---- Middleware: Rate Limiting (SEC-05) ----
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting."""

    _instances: list["RateLimitMiddleware"] = []

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._instances.append(self)

    async def dispatch(self, request: Request, call_next: NextCall):
        if request.url.path in ("/health", "/", "/v1/detect/frame"):
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - settings.rate_window
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]
        if len(self.requests[client_ip]) >= settings.rate_limit:
            return JSONResponse(
                {"error": "Rate limit exceeded. Try again later."},
                status_code=429,
            )
        self.requests[client_ip].append(now)
        return await call_next(request)

    @classmethod
    def reset_all(cls) -> None:
        """Reset rate limit state for all instances (used in tests)."""
        for inst in cls._instances:
            inst.requests.clear()
        cls._instances.clear()


# ---- Middleware: Request ID (MON-01/MON-02/MON-05) ----
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next: NextCall):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        logger.info(
            "Request started: %s %s [request_id=%s]",
            request.method,
            request.url.path,
            request_id,
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---- Middleware: Metrics (MON-03) ----
class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect per-route request count and latency."""

    async def dispatch(self, request: Request, call_next: NextCall):
        start = time.time()
        route = request.url.path
        response = await call_next(request)
        latency = time.time() - start
        REQUEST_COUNT[route] += 1
        REQUEST_LATENCY[route].append(latency)
        # Keep only last 1000 latency measurements per route
        if len(REQUEST_LATENCY[route]) > 1000:
            REQUEST_LATENCY[route] = REQUEST_LATENCY[route][-1000:]
        return response


# ---- Metrics endpoint ----
async def metrics_endpoint() -> Response:
    """Prometheus-compatible metrics endpoint."""
    lines = [
        "# HELP cortex_vision_request_count Total request count by route",
        "# TYPE cortex_vision_request_count counter",
    ]
    for route, count in sorted(REQUEST_COUNT.items()):
        lines.append(f'cortex_vision_request_count{{route="{route}"}} {count}')

    lines.append("# HELP cortex_vision_request_latency_seconds Request latency by route")
    lines.append("# TYPE cortex_vision_request_latency_seconds gauge")
    for route, latencies in sorted(REQUEST_LATENCY.items()):
        if latencies:
            avg = statistics.mean(latencies)
            lines.append(
                f'cortex_vision_request_latency_seconds{{route="{route}",quantile="avg"}} {avg:.4f}'
            )
            if len(latencies) >= 2:
                p99 = sorted(latencies)[int(len(latencies) * 0.99)]
                lines.append(
                    f'cortex_vision_request_latency_seconds{{route="{route}",quantile="p99"}} {p99:.4f}'
                )

    return Response(content="\n".join(lines) + "\n", media_type="text/plain")
