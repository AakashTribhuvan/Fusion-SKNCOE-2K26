from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.services.spoof_detector_service import SpoofDetectorService
from app.services.transcription_service import TranscriptionService
from app.services.vad_service import VADService

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    spoof = SpoofDetectorService.status()
    vad = VADService.status()
    transcription = TranscriptionService.status(settings.whisper_model_size)
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.env,
        "model_status": {
            "speech_detection": vad["status"],
            "transcription": transcription["status"],
            "spoof_detector": spoof["status"],
        },
    }


@router.get("/api/v1/models/status")
def model_status() -> dict:
    spoof = SpoofDetectorService.status()
    vad = VADService.status()
    transcription = TranscriptionService.status(settings.whisper_model_size)
    return {
        "speech_detection": {
            **vad,
        },
        "transcription": {
            **transcription,
        },
        "spoof_detector": {
            "status": spoof["status"],
            "model_name": spoof.get("model_name"),
            "model_version": spoof.get("model_version"),
            "details": spoof.get("reason"),
        },
    }
