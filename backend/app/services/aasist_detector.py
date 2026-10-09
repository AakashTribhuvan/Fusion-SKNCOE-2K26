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


def _prepare_audio(audio: np.ndarray) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        raise ValueError("AASIST requires non-empty audio.")
    if mono.size >= MAX_MODEL_SAMPLES:
        return mono[:MAX_MODEL_SAMPLES]
    # Matches the official evaluation preprocessing for clips shorter than 64600 samples.
    repeats = (MAX_MODEL_SAMPLES // mono.size) + 1
    return np.tile(mono, repeats)[:MAX_MODEL_SAMPLES]


class AASISTDetector:
    """CPU inference adapter for the official pretrained AASIST ASVspoof2019-LA checkpoint."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model = Model(MODEL_CONFIG)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        started = time.perf_counter()
        if sample_rate != MODEL_SAMPLE_RATE:
            return {
                "status": "error",
                "model_name": "AASIST",
                "model_version": AASIST_REVISION,
                "predicted_class": None,
                "genuine_score": None,
                "spoof_score": None,
                "processing_duration_ms": 0,
                "reason": f"AASIST requires {MODEL_SAMPLE_RATE} Hz mono audio; received {sample_rate} Hz.",
            }

        prepared = _prepare_audio(audio)
        waveform = torch.from_numpy(prepared).unsqueeze(0)
        with torch.inference_mode():
            _, logits = self.model(waveform)
            scores = torch.softmax(logits, dim=1)[0]

        spoof_score = float(scores[0].item())
        genuine_score = float(scores[1].item())
        return {
            "status": "completed",
            "model_name": "AASIST",
            "model_version": AASIST_REVISION,
            "model_type": "pretrained_AASIST_ASVspoof2019_LA",
            "predicted_class": "genuine" if genuine_score >= spoof_score else "spoof",
            "genuine_score": genuine_score,
            "spoof_score": spoof_score,
            "score_semantics": "Softmax-normalized AASIST logits for bona fide (class 1) and spoof (class 0); not calibrated probabilities.",
            "threshold": None,
            "processing_duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "checkpoint_sha256": self.checkpoint_sha256,
            "reason": "Experimental pretrained detector. Reported benchmark is ASVspoof 2019 Logical Access; output is not identity verification or a production guarantee.",
        }
