"""Application configuration via environment variables."""

from pydantic import ValidationError
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    api_key: str  # Required — raises RuntimeError if API_KEY env var is missing
    cors_origins: str = "http://localhost:8000"
    upload_dir: str = "uploads"
    output_dir: str = "output"
    yolo_model: str = "yolov8n.pt"
    rate_limit: int = 30
    rate_window: int = 60
    max_batch_results: int = 1000

    model_config = {"env_prefix": ""}


try:
    settings = Settings()
except ValidationError as exc:
    raise RuntimeError(
        "API_KEY environment variable is required — set it or create a .env file."
    ) from exc
