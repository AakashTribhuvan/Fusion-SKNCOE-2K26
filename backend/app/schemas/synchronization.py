from __future__ import annotations

from pydantic import BaseModel, Field


class VideoMouthFrame(BaseModel):
    timestamp_seconds: float = Field(ge=0)
    mouth_motion: float


class VerificationRequest(BaseModel):
    video_frames: list[VideoMouthFrame] = Field(default_factory=list)
