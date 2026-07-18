# =============================================================================
# Stage 1: Build dependencies in a virtual environment
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# System dependencies for OpenCV and YOLO at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in a single layer
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Minimal runtime image
# =============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# System dependencies for OpenCV at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p uploads output

# Create a non-root user for security
RUN groupadd -r cortex && useradd -r -g cortex -d /app -s /sbin/nologin cortex \
    && chown -R cortex:cortex /app
USER cortex

EXPOSE 8000

# Docker health check — enables auto-restart and orchestration monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "src.main"]
