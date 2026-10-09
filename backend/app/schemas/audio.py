from __future__ import annotations

from pydantic import BaseModel, Field


class AudioUploadResponse(BaseModel):
    session_id: str
    recording_id: str
    size_bytes: int
    duration_seconds: float
    status: str = "uploaded"


class AudioQualityReport(BaseModel):
    status: str
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    samples: int | None = None
    peak_amplitude: float | None = None
    rms_energy: float | None = None
    silent: bool | None = None
    clipped: bool | None = None
    speech_usable: bool | None = None
    reason: str | None = None
