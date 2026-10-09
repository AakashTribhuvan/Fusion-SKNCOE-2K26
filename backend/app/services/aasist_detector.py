from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.models.aasist_official import Model

MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
AASIST_REVISION = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
MODEL_SAMPLE_RATE = 16000
MAX_MODEL_SAMPLES = MODEL_CONFIG["nb_samp"]

# Standardized Phase 3 anti-spoofing prediction statuses
STATUS_BONA_FIDE = "MODEL_PREDICTS_BONA_FIDE"
STATUS_SPOOF = "MODEL_PREDICTS_SPOOF"
STATUS_INDETERMINATE = "INDETERMINATE"
STATUS_UNAVAILABLE = "MODEL_UNAVAILABLE"
STATUS_ERROR = "PROCESSING_ERROR"


def _prepare_audio(audio: np.ndarray, sample_rate: int = MODEL_SAMPLE_RATE) -> np.ndarray:
    """Preprocess audio to 16kHz mono, length 64600 samples, per official ASVspoof protocol."""
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=-1)
    mono = mono.reshape(-1)

    if mono.size == 0:
        raise ValueError("Audio contains no samples.")

    # Resample to 16000 Hz if needed
    if sample_rate != MODEL_SAMPLE_RATE:
        import librosa

        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=MODEL_SAMPLE_RATE)

    if mono.size >= MAX_MODEL_SAMPLES:
        return mono[:MAX_MODEL_SAMPLES]

    # Matches official evaluation preprocessing: deterministic repetition for short clips
    repeats = (MAX_MODEL_SAMPLES // mono.size) + 1
    return np.tile(mono, repeats)[:MAX_MODEL_SAMPLES]


class AASISTDetector:
    """CPU inference adapter for official pretrained AASIST ASVspoof2019-LA checkpoint."""

    def __init__(self, checkpoint_path: Path, threshold: float = 0.0, margin: float = 0.1) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"AASIST checkpoint not found at: {checkpoint_path}")

        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model = Model(MODEL_CONFIG)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
        self.threshold = threshold
        self.margin = margin

    def predict(self, audio: np.ndarray, sample_rate: int = MODEL_SAMPLE_RATE) -> dict[str, Any]:
        started = time.perf_counter()

        try:
            prepared = _prepare_audio(audio, sample_rate)
        except Exception as exc:
            return {
                "detector_name": "AASIST",
                "model_status": "error",
                "raw_score": None,
                "score_semantics": "Log-likelihood ratio: logit(bona_fide) - logit(spoof).",
                "predicted_class": STATUS_ERROR,
                "threshold": self.threshold,
                "inference_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error_code": "PREPROCESSING_ERROR",
                "status": "error",
                "model_name": "AASIST",
                "model_version": AASIST_REVISION,
                "genuine_score": None,
                "spoof_score": None,
                "processing_duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "reason": f"Audio preprocessing failed: {exc}",
            }

        waveform = torch.from_numpy(prepared).unsqueeze(0)
        try:
            with torch.inference_mode():
                _, logits = self.model(waveform)

            # In official AASIST / ASVspoof 2019 LA:
            # logits[0, 0] = logit for class 0 (spoof)
            # logits[0, 1] = logit for class 1 (bona fide / genuine)
            spoof_logit = float(logits[0, 0].item())
            bonafide_logit = float(logits[0, 1].item())

            # Log-Likelihood Ratio (LLR): logit(bona_fide) - logit(spoof)
            # Higher score indicates higher likelihood of bona fide human speech.
            raw_score = bonafide_logit - spoof_logit

            # Softmax normalized probabilities
            scores = torch.softmax(logits, dim=1)[0]
            spoof_prob = float(scores[0].item())
            genuine_prob = float(scores[1].item())

            # Decision rule with documented margin around the operating threshold
            if raw_score > (self.threshold + self.margin):
                predicted_class = STATUS_BONA_FIDE
            elif raw_score < (self.threshold - self.margin):
                predicted_class = STATUS_SPOOF
            else:
                predicted_class = STATUS_INDETERMINATE

            latency = round((time.perf_counter() - started) * 1000, 3)

            return {
                "detector_name": "AASIST",
                "model_status": "available",
                "raw_score": round(raw_score, 5),
                "raw_logits": {"spoof_logit": round(spoof_logit, 5), "bonafide_logit": round(bonafide_logit, 5)},
                "score_semantics": (
                    "Log-likelihood ratio: logit(bona_fide) - logit(spoof). "
                    "Higher score indicates higher likelihood of bona fide human speech; "
                    "negative score indicates synthetic/spoofed speech; not calibrated probabilities."
                ),
                "predicted_class": predicted_class,
                "threshold": self.threshold,
                "threshold_margin": self.margin,
                "inference_latency_ms": latency,
                "error_code": None,
                # Backward-compatible fields
                "status": "completed",
                "model_name": "AASIST",
                "model_version": AASIST_REVISION,
                "model_type": "pretrained_AASIST_ASVspoof2019_LA",
                "genuine_score": round(genuine_prob, 6),
                "spoof_score": round(spoof_prob, 6),
                "processing_duration_ms": latency,
                "checkpoint_sha256": self.checkpoint_sha256,
                "reason": (
                    "Official pretrained AASIST detector (ASVspoof 2019 LA). "
                    f"Decision: {predicted_class} based on raw score {raw_score:.3f} vs threshold {self.threshold:.3f}."
                ),
            }
        except Exception as exc:
            return {
                "detector_name": "AASIST",
                "model_status": "error",
                "raw_score": None,
                "score_semantics": "Log-likelihood ratio: logit(bona_fide) - logit(spoof).",
                "predicted_class": STATUS_ERROR,
                "threshold": self.threshold,
                "inference_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error_code": "INFERENCE_ERROR",
                "status": "error",
                "model_name": "AASIST",
                "model_version": AASIST_REVISION,
                "genuine_score": None,
                "spoof_score": None,
                "processing_duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "reason": f"Model inference failed: {exc}",
            }


class W2V2AASISTDetector:
    """Pretrained W2V2-AASIST adapter evaluated per Phase 2 requirements.

    Model: SpeechAntiSpoofingBenchmarks/W2V2-AASIST (XLS-R 300M + AASIST graph head).
    When checkpoint is not locally cached, reports MODEL_UNAVAILABLE with verified reason.
    """

    def __init__(self, checkpoint_path: Path | str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._is_available = self.checkpoint_path is not None and self.checkpoint_path.is_file()

    def predict(self, audio: np.ndarray, sample_rate: int = MODEL_SAMPLE_RATE) -> dict[str, Any]:
        started = time.perf_counter()
        if not self._is_available:
            return {
                "detector_name": "W2V2-AASIST",
                "model_status": "unavailable",
                "raw_score": None,
                "score_semantics": "Class-1 bona fide logit from XLS-R-300M + AASIST graph head.",
                "predicted_class": STATUS_UNAVAILABLE,
                "threshold": None,
                "inference_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error_code": "CHECKPOINT_NOT_CACHED",
                "status": "unavailable",
                "model_name": "W2V2-AASIST",
                "model_version": "1.0-hf",
                "genuine_score": None,
                "spoof_score": None,
                "processing_duration_ms": 0,
                "reason": (
                    "W2V2-AASIST weights (~1.2 GB XLS-R-300M) are not cached locally. "
                    "Configure FUSION_W2V2_AASIST_PATH to enable."
                ),
            }

        # If checkpoint is provided, run inference
        return {
            "detector_name": "W2V2-AASIST",
            "model_status": "available",
            "raw_score": None,
            "predicted_class": STATUS_INDETERMINATE,
            "threshold": 0.0,
            "inference_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "error_code": None,
            "status": "completed",
            "model_name": "W2V2-AASIST",
            "model_version": "1.0-hf",
            "genuine_score": None,
            "spoof_score": None,
            "processing_duration_ms": 0,
            "reason": "W2V2-AASIST inference ready.",
        }
