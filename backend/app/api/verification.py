from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.security import is_session_expired
from app.schemas.synchronization import VerificationRequest
from app.services.audio_decode_service import AudioDecodeService
from app.services.audio_quality_service import AudioQualityService
from app.services.feature_extraction_service import FeatureExtractionService
from app.services.phrase_match_service import PhraseMatchService
from app.services.result_service import ResultService
from app.services.spoof_detector_service import SpoofDetectorService
from app.services.synchronization_service import SynchronizationService
from app.services.transcription_service import TranscriptionService
from app.services.vad_service import VADService

from .sessions import SESSIONS

router = APIRouter(prefix="", tags=["verification"])


@router.post("/api/v1/sessions/{session_id}/verify", response_model=dict)
def verify_session(session_id: str, request: VerificationRequest | None = None) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if is_session_expired(session["created_at"], settings.session_ttl_seconds):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Session expired")
    if session.get("submitted"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A result already exists for this session")
    recording = session.get("recording")
    if recording is None or not recording.get("bytes"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No audio recording has been uploaded for this session")

    started_at = time.perf_counter()
    audio_bytes = recording["bytes"]

    try:
        audio, sample_rate = AudioDecodeService.decode_bytes(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    quality = AudioQualityService.analyze(audio, sample_rate)

    # Only a hard quality failure (empty, too short, or truly silent) blocks
    # downstream analysis. "suspicious" (clipped) audio still contains speech
    # and must flow through VAD and transcription.
    quality_hard_fail = quality.get("status") == "fail"

    if quality_hard_fail:
        speech_detection = {
            "status": quality.get("status"),
            "speech_detected": False,
            "speech_segments": [],
            "speech_duration_seconds": 0.0,
            "reason": quality.get("reason"),
        }
    else:
        speech_detection = VADService().detect(audio, sample_rate)

    should_transcribe = not quality_hard_fail and not (
        speech_detection.get("status") == "completed"
        and not speech_detection.get("speech_detected", False)
    )
    if should_transcribe:
        transcript = TranscriptionService(model_size=settings.whisper_model_size, language=settings.whisper_language).transcribe(audio, sample_rate)
        phrase_result = PhraseMatchService.compare(session["challenge_phrase"], transcript.get("recognized_phrase"))
    else:
        phrase_result = PhraseMatchService.compare(session["challenge_phrase"], None)
        transcript = {
            "status": "incomplete",
            "recognized_phrase": None,
            "language": None,
            "reason": "Audio quality failed or VAD completed with no speech detected.",
        }

    feature_result = FeatureExtractionService.extract(audio, sample_rate)
    if not quality_hard_fail:
        spoof_result = SpoofDetectorService().run(audio, sample_rate)
    else:
        spoof_result = {
            "status": "incomplete",
            "model_name": None,
            "model_version": None,
            "predicted_class": None,
            "genuine_score": None,
            "spoof_score": None,
            "processing_duration_ms": 0,
            "reason": "Spoof analysis was skipped because audio quality checks failed; record clear speech first.",
        }
    video_frames = [frame.model_dump() for frame in request.video_frames] if request else []
    sync_result = SynchronizationService.analyze(
        audio_segments=speech_detection.get("speech_segments"),
        video_frames=video_frames,
    )

    result = ResultService.build(
        session_id=session_id,
        audio_quality=quality,
        speech_detection=speech_detection,
        phrase_verification={
            **phrase_result,
            "expected_phrase": session["challenge_phrase"],
            "recognized_phrase": transcript.get("recognized_phrase") or phrase_result.get("recognized_phrase"),
            "language": transcript.get("language"),
            "transcription_status": transcript.get("status"),
        },
        feature_extraction=feature_result,
        spoof_detection=spoof_result,
        audio_video_sync=sync_result,
        started_at=started_at,
    )

    session["result"] = result
    session["submitted"] = True
    return result
