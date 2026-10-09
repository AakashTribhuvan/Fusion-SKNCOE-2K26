from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.security import generate_session_id, is_session_expired, utc_now, utc_ts
from app.services.audio_decode_service import AudioDecodeService
from app.services.audio_quality_service import AudioQualityService
from app.services.challenge_service import ChallengeService

router = APIRouter(prefix="/sessions", tags=["sessions"])

SESSIONS: dict[str, dict[str, Any]] = {}


@router.post("", response_model=dict)
def create_session() -> dict[str, Any]:
    session_id = generate_session_id()
    challenge = ChallengeService.create(ttl_seconds=settings.session_ttl_seconds)
    expires_at = challenge.expires_at
    SESSIONS[session_id] = {
        "session_id": session_id,
        "challenge_phrase": challenge.phrase,
        "created_at": challenge.created_at,
        "expires_at": expires_at,
        "previous_phrase": None,
        "recording": None,
        "result": None,
        "submitted": False,
    }
    return {
        "session_id": session_id,
        "challenge_phrase": challenge.phrase,
        "created_at": utc_ts(challenge.created_at),
        "expires_at": utc_ts(expires_at),
    }


@router.post("/{session_id}/challenge", response_model=dict)
def replace_challenge(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if is_session_expired(session["created_at"], settings.session_ttl_seconds):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Session expired")
    previous_phrase = session["challenge_phrase"]
    new_challenge = ChallengeService.create(ttl_seconds=settings.session_ttl_seconds)
    session["previous_phrase"] = previous_phrase
    session["challenge_phrase"] = new_challenge.phrase
    session["created_at"] = new_challenge.created_at
    session["expires_at"] = new_challenge.expires_at
    session["submitted"] = False
    return {
        "session_id": session_id,
        "challenge_phrase": new_challenge.phrase,
        "created_at": utc_ts(new_challenge.created_at),
        "expires_at": utc_ts(new_challenge.expires_at),
        "invalidated_previous": True,
    }


@router.post("/{session_id}/audio", response_model=dict)
def upload_audio(session_id: str, file: UploadFile) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if is_session_expired(session["created_at"], settings.session_ttl_seconds):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Session expired")
    if file is None or file.filename is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is required")

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Recording is empty")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Upload exceeds configured size limit")
    try:
        decoded_audio, sample_rate = AudioDecodeService.decode_bytes(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Audio could not be decoded. Submit a valid PCM WAV or supported WebM/Opus/MP4/Ogg recording.",
        ) from exc

    quality = AudioQualityService.analyze(decoded_audio, sample_rate)
    duration_seconds = quality.get("duration_seconds", 0.0)
    if duration_seconds < 0.3:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Recording is too short; record at least 0.3 seconds.")
    if duration_seconds > settings.max_audio_seconds:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Recording exceeds the {settings.max_audio_seconds}-second limit.",
        )

    session["recording"] = {
        "filename": file.filename,
        "content_type": file.content_type,
        "bytes": raw,
        "size_bytes": len(raw),
        "duration_seconds": duration_seconds,
    }

    return {
        "session_id": session_id,
        "recording_id": f"rec_{session_id}",
        "size_bytes": len(raw),
        "duration_seconds": duration_seconds,
        "status": "uploaded",
    }


@router.get("/{session_id}/result", response_model=dict)
def get_result(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if is_session_expired(session["created_at"], settings.session_ttl_seconds):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Session expired")
    if session["result"] is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No verification result available yet")
    return session["result"]
