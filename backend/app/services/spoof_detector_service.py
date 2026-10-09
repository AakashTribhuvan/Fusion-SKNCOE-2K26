from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.spoof_detector_interface import AudioSpoofDetector
from app.services.aasist_detector import (
    AASISTDetector,
    AASIST_REVISION,
    STATUS_BONA_FIDE,
    STATUS_SPOOF,
    STATUS_INDETERMINATE,
    STATUS_UNAVAILABLE,
    STATUS_ERROR,
    W2V2AASISTDetector,
)
from app.training.baseline import LogisticBaseline
from app.training.features import extract_feature_vector


class TrainedBaselineDetector(AudioSpoofDetector):
    def __init__(self, checkpoint_path: Path) -> None:
        self.model, self.metadata = LogisticBaseline.from_file(checkpoint_path)

    def predict(self, audio, sample_rate: int) -> dict[str, Any]:
        started = time.perf_counter()
        if sample_rate != self.metadata["sample_rate"]:
            latency = round((time.perf_counter() - started) * 1000, 3)
            return {
                "detector_name": self.metadata.get("model_name", "LogisticBaseline"),
                "model_status": "error",
                "raw_score": None,
                "score_semantics": self.metadata.get("spoof_score_semantics", "probability of spoof"),
                "predicted_class": STATUS_ERROR,
                "threshold": self.model.threshold,
                "inference_latency_ms": latency,
                "error_code": "SAMPLE_RATE_MISMATCH",
                "status": "error",
                "model_name": self.metadata["model_name"],
                "model_version": self.metadata["model_version"],
                "genuine_score": None,
                "spoof_score": None,
                "processing_duration_ms": latency,
                "reason": f"Model requires {self.metadata['sample_rate']} Hz audio; received {sample_rate} Hz.",
            }
        features = extract_feature_vector(audio, sample_rate)[None, :]
        spoof_score = float(self.model.predict_spoof_scores(features)[0])
        threshold = self.model.threshold
        is_spoof = spoof_score >= threshold
        latency = round((time.perf_counter() - started) * 1000, 3)

        return {
            "detector_name": self.metadata.get("model_name", "LogisticBaseline"),
            "model_status": "available",
            "raw_score": spoof_score,
            "score_semantics": self.metadata.get("spoof_score_semantics", "logistic probability of spoof"),
            "predicted_class": STATUS_SPOOF if is_spoof else STATUS_BONA_FIDE,
            "threshold": threshold,
            "inference_latency_ms": latency,
            "error_code": None,
            "status": "completed",
            "model_name": self.metadata["model_name"],
            "model_version": self.metadata["model_version"],
            "model_type": self.metadata["model_type"],
            "genuine_score": round(1.0 - spoof_score, 6),
            "spoof_score": round(spoof_score, 6),
            "processing_duration_ms": latency,
            "reason": self.metadata.get("training_metadata", {}).get(
                "warning",
                "Research baseline only; not a production security guarantee.",
            ),
        }


class UnavailableSpoofDetector(AudioSpoofDetector):
    def predict(self, audio, sample_rate: int) -> dict[str, Any]:
        return {
            "detector_name": "none",
            "model_status": "unavailable",
            "raw_score": None,
            "score_semantics": "No model loaded.",
            "predicted_class": STATUS_UNAVAILABLE,
            "threshold": None,
            "inference_latency_ms": 0.0,
            "error_code": "MODEL_NOT_CONFIGURED",
            "status": "unavailable",
            "model_name": None,
            "model_version": None,
            "genuine_score": None,
            "spoof_score": None,
            "processing_duration_ms": 0,
            "reason": "Compatible anti-spoofing checkpoint is not configured.",
        }


@lru_cache(maxsize=1)
def _load_aasist_detector(checkpoint_path: str) -> AASISTDetector:
    return AASISTDetector(Path(checkpoint_path))


class SpoofDetectorService:
    """Centralized anti-spoofing verification service."""

    def __init__(self, detector: AudioSpoofDetector | None = None) -> None:
        self.detector = detector or UnavailableSpoofDetector()

    def run(self, audio, sample_rate: int) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            if not isinstance(self.detector, UnavailableSpoofDetector):
                return self.detector.predict(audio, sample_rate)

            if not settings.spoof_detector_enabled:
                return {
                    **UnavailableSpoofDetector().predict(audio, sample_rate),
                    "model_status": "disabled",
                    "reason": "Spoof detector is disabled by configuration.",
                }

            # Multi-Model Ensemble with Heuristic Risk Fusion
            from app.services.ensemble_service import get_ensemble_service

            ens = get_ensemble_service().evaluate(audio, sample_rate)
            spoof_risk = ens.get("ensemble_spoof_risk")
            decision_band = ens.get("decision_band")
            if decision_band == "HIGH_SPOOF_RISK_REVIEW":
                predicted_class = STATUS_SPOOF
            elif decision_band == "LOWER_MODEL_ESTIMATED_SPOOF_RISK":
                predicted_class = STATUS_BONA_FIDE
            elif decision_band == "MODEL_UNAVAILABLE":
                predicted_class = STATUS_UNAVAILABLE
            else:
                predicted_class = STATUS_INDETERMINATE

            return {
                "detector_name": "Ensemble (W2V2-AASIST + AASIST + Wav2Vec2 + Heuristics)",
                "model_status": "available" if not ens.get("is_degraded") else "degraded",
                "raw_score": spoof_risk,
                "score_semantics": (
                    "Provisional weighted ensemble spoof-risk index (0-100 scale). "
                    "Uncalibrated composite of AASIST, Wav2Vec2 deepfake detector, and signal heuristics."
                ),
                "predicted_class": predicted_class,
                "decision_band": decision_band,
                "ensemble_spoof_risk": spoof_risk,
                "threshold": ens.get("thresholds", {}).get("high_risk_threshold", 75.0),
                "review_threshold": ens.get("thresholds", {}).get("review_threshold", 70.0),
                "high_risk_threshold": ens.get("thresholds", {}).get("high_risk_threshold", 75.0),
                "inference_latency_ms": ens.get("latency_ms", 0.0),
                "error_code": None,
                "status": "completed",
                "model_name": "Three-Model-Ensemble",
                "model_version": "1.0-fusion",
                "genuine_score": round((100.0 - spoof_risk) / 100.0, 4) if spoof_risk is not None else None,
                "spoof_score": round(spoof_risk / 100.0, 4) if spoof_risk is not None else None,
                "processing_duration_ms": ens.get("latency_ms", 0.0),
                "is_degraded": ens.get("is_degraded", False),
                "missing_models": ens.get("missing_models", []),
                "degraded_reasons": ens.get("degraded_reasons", []),
                "effective_weights": ens.get("effective_weights", {}),
                "models": ens.get("models", {}),
                "heuristics": ens.get("heuristics", {}),
                "decision_explanation": ens.get("decision_explanation", ""),
                "reason": ens.get("decision_explanation", ""),
            }

        except Exception as exc:
            latency = round((time.perf_counter() - started) * 1000, 3)
            return {
                "detector_name": settings.spoof_model_name or "AASIST",
                "model_status": "error",
                "raw_score": None,
                "score_semantics": "Inference exception occurred.",
                "predicted_class": STATUS_ERROR,
                "threshold": None,
                "inference_latency_ms": latency,
                "error_code": "INFERENCE_EXCEPTION",
                "status": "error",
                "model_name": settings.spoof_model_name,
                "model_version": None,
                "genuine_score": None,
                "spoof_score": None,
                "processing_duration_ms": latency,
                "reason": f"Spoof detection failed: {exc}",
            }

    @staticmethod
    def status() -> dict[str, Any]:
        if not settings.spoof_detector_enabled:
            return {
                "status": "disabled",
                "model_name": None,
                "model_version": None,
                "reason": "Spoof detector is disabled by configuration.",
            }

        # Check official AASIST
        if settings.aasist_model_path.is_file():
            try:
                detector = _load_aasist_detector(str(settings.aasist_model_path.resolve()))
                return {
                    "status": "available",
                    "model_name": "AASIST",
                    "model_version": AASIST_REVISION,
                    "model_type": "pretrained_AASIST_ASVspoof2019_LA",
                    "checkpoint_sha256": detector.checkpoint_sha256,
                    "threshold": detector.threshold,
                    "margin": detector.margin,
                    "reason": "Pretrained AASIST detector loaded and verified for CPU inference.",
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "model_name": "AASIST",
                    "model_version": None,
                    "reason": f"AASIST checkpoint could not be loaded: {exc}",
                }

        # Check custom checkpoint
        if settings.spoof_model_path:
            checkpoint_path = Path(settings.spoof_model_path)
            if not checkpoint_path.is_file():
                return {
                    "status": "unavailable",
                    "model_name": None,
                    "model_version": None,
                    "reason": "Configured checkpoint does not exist.",
                }
            try:
                detector = TrainedBaselineDetector(checkpoint_path)
                return {
                    "status": "available",
                    "model_name": detector.metadata["model_name"],
                    "model_version": detector.metadata["model_version"],
                    "model_type": detector.metadata["model_type"],
                    "reason": detector.metadata.get("training_metadata", {}).get("warning"),
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "model_name": None,
                    "model_version": None,
                    "reason": f"Checkpoint could not be loaded: {exc}",
                }

        return {
            "status": "unavailable",
            "model_name": None,
            "model_version": None,
            "reason": "No configured spoof checkpoint was found.",
        }
