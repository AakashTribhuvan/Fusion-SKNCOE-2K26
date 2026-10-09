from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


@dataclass
class Settings:
    app_name: str = "FUSION"
    env: str = os.getenv("FUSION_ENV", "development")
    api_prefix: str = os.getenv("FUSION_API_PREFIX", "/api/v1")
    session_ttl_seconds: int = int(os.getenv("FUSION_SESSION_TTL_SECONDS", "900"))
    max_audio_seconds: int = int(os.getenv("FUSION_MAX_AUDIO_SECONDS", "10"))
    max_upload_bytes: int = int(os.getenv("FUSION_MAX_UPLOAD_BYTES", "10485760"))
    spoof_model_path: str | None = os.getenv("FUSION_SPOOF_MODEL_PATH") or None
    spoof_model_name: str = os.getenv("FUSION_SPOOF_MODEL_NAME", "unavailable")
    spoof_detector_enabled: bool = os.getenv("FUSION_SPOOF_DETECTOR_ENABLED", "true").lower() == "true"
    aasist_model_path: Path = Path(
        os.getenv("FUSION_AASIST_MODEL_PATH")
        or str(Path(__file__).resolve().parents[1] / "models" / "weights" / "AASIST.pth")
    )
    whisper_model_size: str = os.getenv("FUSION_WHISPER_MODEL_SIZE", "tiny")
    whisper_language: str = os.getenv("FUSION_WHISPER_LANGUAGE", "en")
    sample_rate: int = int(os.getenv("FUSION_AUDIO_SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("FUSION_AUDIO_CHANNELS", "1"))
    host: str = os.getenv("FUSION_HOST", "0.0.0.0")
    port: int = int(os.getenv("FUSION_PORT", "8000"))
    temp_dir: Path = Path(os.getenv("FUSION_TEMP_DIR", "/tmp/fusion"))


settings = Settings()
