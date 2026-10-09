from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.spoof_detector_interface import AudioSpoofDetector
from app.services.aasist_detector import AASISTDetector, AASIST_REVISION
from app.training.baseline import LogisticBaseline
from app.training.features import extract_feature_vector


class TrainedBaselineDetector(AudioSpoofDetector):
    def __init__(self, checkpoint_path: Path) -> None:
        self.model, self.metadata = LogisticBaseline.from_file(checkpoint_path)

    def predict(self, audio, sample_rate: int) -> dict[str, Any]:
        started = time.perf_counter()
        if sample_rate != self.metadata["sample_rate"]:
            return {
                "status": "error",
                "model_name": self.metadata["model_name"],
                "model_version": self.metadata["model_version"],
                "predicted_class": None,
                "genuine_score": None,
                "spoof_score": None,
                "processing_duration_ms": 0,
                "reason": f"Model requires {self.metadata['sample_rate']} Hz audio; received {sample_rate} Hz.",
            }
        features = extract_feature_vector(audio, sample_rate)[None, :]
        spoof_score = float(self.model.predict_spoof_scores(features)[0])
        threshold = self.model.threshold
        return {
            "status": "completed",
            "model_name": self.metadata["model_name"],
            "model_version": self.metadata["model_version"],
            "model_type": self.metadata["model_type"],
            "predicted_class": "spoof" if spoof_score >= threshold else "genuine",
            "genuine_score": 1.0 - spoof_score,
            "spoof_score": spoof_score,
            "score_semantics": self.metadata["spoof_score_semantics"],
            "threshold": threshold,
            "processing_duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "reason": self.metadata.get("training_metadata", {}).get(
                "warning",
                "Research baseline only; not a production security guarantee.",
            ),
        }


class UnavailableSpoofDetector(AudioSpoofDetector):
    def predict(self, audio, sample_rate: int):
        return {
            "status": "unavailable",
            "model_name": None,
            "model_version": None,
            "predicted_class": None,
            "genuine_score": None,
            "spoof_score": None,
            "processing_duration_ms": 0,
            "reason": "Compatible checkpoint is not configured.",
        }


@lru_cache(maxsize=1)
def _load_aasist_detector(checkpoint_path: str) -> AASISTDetector:
    return AASISTDetector(Path(checkpoint_path))


class SpoofDetectorService:
    def __init__(self, detector: AudioSpoofDetector | None = None) -> None:
        self.detector = detector or UnavailableSpoofDetector()

    def run(self, audio, sample_rate: int) -> dict[str, Any]:
        try:
            if not isinstance(self.detector, UnavailableSpoofDetector):
                return self.detector.predict(audio, sample_rate)
            if not settings.spoof_detector_enabled:
                return {
                    **UnavailableSpoofDetector().predict(audio, sample_rate),
                    "reason": "Spoof detector is disabled by configuration.",
                }
            if settings.spoof_model_path:
                checkpoint_path = Path(settings.spoof_model_path)
                if not checkpoint_path.is_file():
                    raise FileNotFoundError("Configured model checkpoint does not exist.")
                self.detector = TrainedBaselineDetector(checkpoint_path)
                return self.detector.predict(audio, sample_rate)
            if settings.aasist_model_path.is_file():
                self.detector = _load_aasist_detector(str(settings.aasist_model_path.resolve()))
                return self.detector.predict(audio, sample_rate)
            return {
                **UnavailableSpoofDetector().predict(audio, sample_rate),
                "reason": "No configured spoof checkpoint was found.",
            }
        except Exception as exc:  # pragma: no cover
            return {
                "status": "error",
                "model_name": settings.spoof_model_name,
                "model_version": None,
                "predicted_class": None,
                "genuine_score": None,
                "spoof_score": None,
                "processing_duration_ms": 0,
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
        if not settings.spoof_model_path:
            if not settings.aasist_model_path.is_file():
                return {
                    "status": "unavailable",
                    "model_name": None,
                    "model_version": None,
                    "reason": "No configured spoof checkpoint was found.",
                }
            try:
                detector = _load_aasist_detector(str(settings.aasist_model_path.resolve()))
                return {
                    "status": "available",
                    "model_name": "AASIST",
                    "model_version": AASIST_REVISION,
                    "model_type": "pretrained_AASIST_ASVspoof2019_LA",
                    "checkpoint_sha256": detector.checkpoint_sha256,
                    "reason": "Experimental pretrained detector, benchmarked on ASVspoof 2019 Logical Access.",
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "model_name": "AASIST",
                    "model_version": None,
                    "reason": f"Checkpoint could not be loaded: {exc}",
                }
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
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {
                "status": "error",
                "model_name": None,
                "model_version": None,
                "reason": f"Checkpoint could not be loaded: {exc}",
            }
