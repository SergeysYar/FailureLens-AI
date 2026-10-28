from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path = Path(__file__).resolve().parents[2]
    environment: str = os.getenv("FAILURELENS_ENV", "development")
    model_dir: Path = Path(os.getenv("FAILURELENS_MODEL_DIR", "artifacts"))
    registry_path: Path = Path(os.getenv("FAILURELENS_REGISTRY", "artifacts/registry.json"))
    log_level: str = os.getenv("FAILURELENS_LOG_LEVEL", "INFO")
    allowed_origins: tuple[str, ...] = tuple(filter(None, os.getenv("FAILURELENS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")))


settings = Settings()

