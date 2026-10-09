from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    session_id: str
    audio_quality: dict[str, Any] = Field(default_factory=dict)
    speech_detection: dict[str, Any] = Field(default_factory=dict)
    phrase_verification: dict[str, Any] = Field(default_factory=dict)
    feature_extraction: dict[str, Any] = Field(default_factory=dict)
    spoof_detection: dict[str, Any] = Field(default_factory=dict)
    audio_video_sync: dict[str, Any] = Field(default_factory=dict)
    overall_status: str = "incomplete"
    processing_time_ms: int | None = None
