from __future__ import annotations

import time
from typing import Any

from app.services.aasist_detector import (
    STATUS_BONA_FIDE,
    STATUS_SPOOF,
    STATUS_INDETERMINATE,
    STATUS_UNAVAILABLE,
    STATUS_ERROR,
)


class ResultService:
    @staticmethod
    def build(
        session_id: str,
        audio_quality: dict[str, Any],
        speech_detection: dict[str, Any],
        phrase_verification: dict[str, Any],
        feature_extraction: dict[str, Any],
        spoof_detection: dict[str, Any],
        audio_video_sync: dict[str, Any],
        started_at: float,
    ) -> dict[str, Any]:
        reasons: list[str] = []

        # 1. Usable speech validation
        usable_speech = (
            audio_quality.get("status") in {"pass", "suspicious"}
            and speech_detection.get("status") == "completed"
            and speech_detection.get("speech_detected") is True
        )
        if not usable_speech:
            reasons.append("Audio quality check or voice activity detection failed.")

        # 2. Phrase matching check
        phrase_status = phrase_verification.get("phrase_verification_status") or phrase_verification.get("status")
        exact_match = phrase_verification.get("exact_match")
        phrase_passed = exact_match is True
        phrase_failed = exact_match is False or phrase_status == "fail"
        if phrase_failed:
            reasons.append("Spoken phrase did not match the expected randomized challenge phrase.")

        # 3. Anti-spoofing check
        spoof_pred = spoof_detection.get("decision_band") or spoof_detection.get("predicted_class")
        spoof_detected = spoof_pred in {STATUS_SPOOF, "spoof", "HIGH_SPOOF_RISK_REVIEW"}
        review_band = spoof_pred in {STATUS_INDETERMINATE, "REVIEW_REQUIRED"}
        bona_fide_detected = spoof_pred in {STATUS_BONA_FIDE, "genuine", "LOWER_MODEL_ESTIMATED_SPOOF_RISK"}
        spoof_uncertain = (
            review_band
            or spoof_pred in {STATUS_UNAVAILABLE, STATUS_ERROR, None, "MODEL_UNAVAILABLE", "PROCESSING_ERROR"}
            or spoof_detection.get("status") != "completed"
        )
        if spoof_detected:
            reasons.append("Anti-spoofing detector classified audio as synthetic, cloned, or deepfake speech.")
        elif review_band:
            reasons.append(
                f"Anti-spoofing ensemble returned intermediate review risk ({spoof_pred}); human review required."
            )
        elif spoof_uncertain:
            reasons.append(
                f"Anti-spoofing check is inconclusive or unavailable ({spoof_pred or spoof_detection.get('status')})."
            )

        # 4. Overall status determination
        # A matching phrase MUST NEVER override a spoof failure!
        if spoof_detected or (usable_speech and phrase_failed):
            overall_status = "fail"
            risk_category = "HIGH_RISK"
        elif not usable_speech:
            overall_status = "incomplete"
            risk_category = "INDETERMINATE"
        elif spoof_uncertain:
            overall_status = "review_required"
            risk_category = "REVIEW_REQUIRED"
        elif usable_speech and phrase_passed and bona_fide_detected:
            overall_status = "pass"
            risk_category = "LOW_RISK"
            reasons.append("All independent checks passed: usable speech, phrase matched, and anti-spoofing verified bona fide.")
        else:
            overall_status = "review_required"
            risk_category = "REVIEW_REQUIRED"

        return {
            "session_id": session_id,
            "audio_quality": audio_quality,
            "speech_detection": speech_detection,
            "phrase_verification": phrase_verification,
            "feature_extraction": feature_extraction,
            "spoof_detection": spoof_detection,
            "audio_video_sync": audio_video_sync,
            "overall_status": overall_status,
            "risk_category": risk_category,
            "reasons": reasons,
            "processing_time_ms": int((time.perf_counter() - started_at) * 1000),
        }
