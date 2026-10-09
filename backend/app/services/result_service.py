from __future__ import annotations

import time
from typing import Any


class ResultService:
    @staticmethod
    def build(session_id: str, audio_quality: dict[str, Any], speech_detection: dict[str, Any], phrase_verification: dict[str, Any], feature_extraction: dict[str, Any], spoof_detection: dict[str, Any], audio_video_sync: dict[str, Any], started_at: float) -> dict[str, Any]:
        overall_status = "incomplete"
        usable_speech = (
            audio_quality.get("status") == "pass"
            and speech_detection.get("status") == "completed"
            and speech_detection.get("speech_detected") is True
        )
        phrase_failed = (
            phrase_verification.get("status") == "fail"
            or phrase_verification.get("exact_match") is False
        )
        spoof_failed = (
            spoof_detection.get("status") == "completed"
            and spoof_detection.get("predicted_class") == "spoof"
        )

        if usable_speech and (phrase_failed or spoof_failed):
            overall_status = "fail"
        elif (
            usable_speech
            and phrase_verification.get("exact_match") is True
            and spoof_detection.get("status") == "completed"
            and spoof_detection.get("predicted_class") == "genuine"
        ):
            overall_status = "pass"

        return {
            "session_id": session_id,
            "audio_quality": audio_quality,
            "speech_detection": speech_detection,
            "phrase_verification": phrase_verification,
            "feature_extraction": feature_extraction,
            "spoof_detection": spoof_detection,
            "audio_video_sync": audio_video_sync,
            "overall_status": overall_status,
            "processing_time_ms": int((time.perf_counter() - started_at) * 1000),
        }
