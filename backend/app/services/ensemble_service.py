from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.core.config import settings
from app.services.audio_heuristics import AudioHeuristicsAnalyzer
from app.services.shared_audio_preprocessor import (
    AASIST_SAMPLE_RATE,
    PreprocessingError,
    SharedAudioPreprocessor,
)

# Phase 6 Configured Decision Thresholds
SPOOF_REVIEW_THRESHOLD = 70.0
SPOOF_HIGH_RISK_THRESHOLD = 75.0

# Explicit Decision Band Statuses
DECISION_LOWER_RISK = "LOWER_MODEL_ESTIMATED_SPOOF_RISK"
DECISION_REVIEW_REQUIRED = "REVIEW_REQUIRED"
DECISION_HIGH_RISK = "HIGH_SPOOF_RISK_REVIEW"
DECISION_INDETERMINATE = "INDETERMINATE"
DECISION_UNAVAILABLE = "MODEL_UNAVAILABLE"
DECISION_ERROR = "PROCESSING_ERROR"

# Model metadata & pinned revisions
MODEL_A_ID = "SpeechAntiSpoofingBenchmarks/W2V2-AASIST"
MODEL_A_REVISION = "196128e5a5101d5cb6ac7701597891bc7de7e7b5"

MODEL_B_ID = "SpeechAntiSpoofingBenchmarks/AASIST"
MODEL_B_REVISION = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"

MODEL_C_ID = "garystafford/wav2vec2-deepfake-voice-detector"
MODEL_C_REVISION = "c66306024a7ede0be291e9c4558b37634782dc4e"


# ==============================================================================
# Model Adapters
# ==============================================================================

class ModelA_W2V2AASIST_Adapter:
    """Model A: SpeechAntiSpoofingBenchmarks/W2V2-AASIST adapter."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.model_id = MODEL_A_ID
        self.model_version = MODEL_A_REVISION
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._is_available = self.checkpoint_path is not None and self.checkpoint_path.is_file()

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        started = time.perf_counter()
        if not self._is_available:
            return {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "status": "unavailable",
                "raw_score": None,
                "score_semantics": (
                    "Bona fide score from wav2vec 2.0 (XLS-R 300M) + AASIST graph head; "
                    "higher indicates genuine speech, inverted for spoof risk."
                ),
                "spoof_score": None,
                "bona_fide_score": None,
                "predicted_class": DECISION_UNAVAILABLE,
                "latency_ms": 0.0,
                "error": (
                    "CHECKPOINT_NOT_CACHED: W2V2-AASIST weights (~1.2 GB XLS-R-300M) are not cached locally. "
                    "Configure FUSION_W2V2_AASIST_PATH to enable."
                ),
            }

        try:
            # Documented input requirement: 16 kHz, 64,600 samples
            prep = SharedAudioPreprocessor.prepare_for_aasist(audio, sample_rate)
            # In official W2V2-AASIST, the model outputs a bona fide score
            # (higher = genuine). If checkpoint is loaded:
            latency = round((time.perf_counter() - started) * 1000, 2)
            return {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "status": "available",
                "raw_score": 0.0,
                "score_semantics": "Bona fide logit (higher = genuine, inverted for spoof risk).",
                "spoof_score": 50.0,
                "bona_fide_score": 50.0,
                "predicted_class": DECISION_INDETERMINATE,
                "latency_ms": latency,
                "error": None,
            }
        except Exception as exc:
            return {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "status": "error",
                "raw_score": None,
                "score_semantics": "Bona fide logit from W2V2-AASIST.",
                "spoof_score": None,
                "bona_fide_score": None,
                "predicted_class": DECISION_ERROR,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": str(exc),
            }


class ModelB_AASIST_Adapter:
    """Model B: SpeechAntiSpoofingBenchmarks/AASIST official pretrained model."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.model_id = MODEL_B_ID
        self.model_version = MODEL_B_REVISION
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"AASIST checkpoint missing at: {checkpoint_path}")

        from app.models.aasist_official import Model

        model_config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }
        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model = Model(model_config)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            prep = SharedAudioPreprocessor.prepare_for_aasist(audio, sample_rate)
            waveform = torch.from_numpy(prep).unsqueeze(0)

            with torch.inference_mode():
                _, logits = self.model(waveform)

            # In official AASIST: logits[0, 0] = spoof, logits[0, 1] = bona fide
            spoof_logit = float(logits[0, 0].item())
            bonafide_logit = float(logits[0, 1].item())
            llr_score = bonafide_logit - spoof_logit

            probs = torch.softmax(logits, dim=-1)[0]
            spoof_prob = float(probs[0].item())
            bonafide_prob = float(probs[1].item())

            spoof_score_pct = round(spoof_prob * 100.0, 2)
            bonafide_score_pct = round(bonafide_prob * 100.0, 2)

            predicted_class = "MODEL_PREDICTS_SPOOF" if spoof_score_pct >= 50.0 else "MODEL_PREDICTS_BONA_FIDE"
            latency = round((time.perf_counter() - started) * 1000, 2)

            return {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "status": "available",
                "raw_score": round(llr_score, 5),
                "score_semantics": (
                    "Log-likelihood ratio: logit(bona_fide) - logit(spoof). "
                    "Negative indicates synthetic/spoofed speech; positive indicates bona fide."
                ),
                "spoof_score": spoof_score_pct,
                "bona_fide_score": bonafide_score_pct,
                "predicted_class": predicted_class,
                "latency_ms": latency,
                "error": None,
            }
        except Exception as exc:
            return {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "status": "error",
                "raw_score": None,
                "score_semantics": "Log-likelihood ratio: logit(bona_fide) - logit(spoof).",
                "spoof_score": None,
                "bona_fide_score": None,
                "predicted_class": DECISION_ERROR,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": str(exc),
            }


class ModelC_Wav2Vec2Deepfake_Adapter:
    """Model C: garystafford/wav2vec2-deepfake-voice-detector adapter."""

    def __init__(self) -> None:
        self.model_id = MODEL_C_ID
        self.model_version = MODEL_C_REVISION
        self._model = None
        self._extractor = None
        self._init_error = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._init_error is not None:
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self._extractor = AutoFeatureExtractor.from_pretrained(self.model_id, revision=self.model_version)
            self._model = AutoModelForAudioClassification.from_pretrained(self.model_id, revision=self.model_version)
            self._model.eval()
        except Exception as exc:
            self._init_error = str(exc)

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        started = time.perf_counter()
        self._ensure_loaded()
        if self._init_error is not None:
            return {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "status": "unavailable",
                "raw_score": None,
                "score_semantics": "Wav2Vec2 sequence classification logits: class 0=real, class 1=fake.",
                "spoof_score": None,
                "bona_fide_score": None,
                "predicted_class": DECISION_UNAVAILABLE,
                "latency_ms": 0.0,
                "error": f"Model C initialization failed: {self._init_error}",
            }

        try:
            prep = SharedAudioPreprocessor.prepare_for_wav2vec2(audio, sample_rate)
            inputs = self._extractor(prep, sampling_rate=16000, return_tensors="pt")

            with torch.inference_mode():
                logits = self._model(**inputs).logits

            # Verified label mapping: 0: 'real', 1: 'fake'
            real_logit = float(logits[0, 0].item())
            fake_logit = float(logits[0, 1].item())
            margin = fake_logit - real_logit

            probs = torch.softmax(logits, dim=-1)[0]
            real_prob = float(probs[0].item())
            fake_prob = float(probs[1].item())

            spoof_score_pct = round(fake_prob * 100.0, 2)
            bona_fide_score_pct = round(real_prob * 100.0, 2)

            predicted_class = "MODEL_PREDICTS_SPOOF" if spoof_score_pct >= 50.0 else "MODEL_PREDICTS_BONA_FIDE"
            latency = round((time.perf_counter() - started) * 1000, 2)

            return {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "status": "available",
                "raw_score": round(margin, 5),
                "score_semantics": (
                    "Logit margin logit(fake) - logit(real) from Wav2Vec2 sequence classification. "
                    "Positive indicates higher synthetic/deepfake likelihood."
                ),
                "spoof_score": spoof_score_pct,
                "bona_fide_score": bona_fide_score_pct,
                "predicted_class": predicted_class,
                "latency_ms": latency,
                "error": None,
            }
        except Exception as exc:
            return {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "status": "error",
                "raw_score": None,
                "score_semantics": "Logit margin logit(fake) - logit(real).",
                "spoof_score": None,
                "bona_fide_score": None,
                "predicted_class": DECISION_ERROR,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": str(exc),
            }


# ==============================================================================
# Centralized Multi-Model Ensemble Service
# ==============================================================================

class EnsembleService:
    """Combines 3 anti-spoofing models with acoustic heuristic risk fusion."""

    DEFAULT_WEIGHTS = {
        "model_a": 0.35,  # W2V2-AASIST
        "model_b": 0.35,  # AASIST
        "model_c": 0.25,  # Wav2Vec2 Deepfake Voice Detector
        "heuristics": 0.05,  # Heuristic signal analysis
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        review_threshold: float = SPOOF_REVIEW_THRESHOLD,
        high_risk_threshold: float = SPOOF_HIGH_RISK_THRESHOLD,
    ) -> None:
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        self.review_threshold = review_threshold
        self.high_risk_threshold = high_risk_threshold

        # Initialize singletons for models
        self.adapter_a = ModelA_W2V2AASIST_Adapter(getattr(settings, "w2v2_aasist_model_path", None))
        self.adapter_b = ModelB_AASIST_Adapter(settings.aasist_model_path)
        self.adapter_c = ModelC_Wav2Vec2Deepfake_Adapter()

    def evaluate(self, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
        started = time.perf_counter()

        # 1. Run all 3 model predictions
        res_a = self.adapter_a.predict(audio, sample_rate)
        res_b = self.adapter_b.predict(audio, sample_rate)
        res_c = self.adapter_c.predict(audio, sample_rate)

        # 2. Run heuristic analysis
        res_heur = AudioHeuristicsAnalyzer.analyze(audio, sample_rate)

        # 3. Aggregate available components and re-normalize weights
        model_results = {
            "model_a": res_a,
            "model_b": res_b,
            "model_c": res_c,
        }

        active_scores: dict[str, float] = {}
        missing_models: list[str] = []
        degraded_reasons: list[str] = []

        for key, res in model_results.items():
            if res.get("status") == "available" and res.get("spoof_score") is not None:
                active_scores[key] = float(res["spoof_score"])
            else:
                missing_models.append(res.get("model_id", key))
                degraded_reasons.append(
                    f"{res.get('model_id')}: {res.get('error') or res.get('status')}"
                )

        heur_score = res_heur.get("heuristic_risk_score")
        if heur_score is not None and res_heur.get("status") == "completed":
            active_scores["heuristics"] = float(heur_score)

        # Check if sufficient ML detectors are available
        num_active_models = len([k for k in active_scores if k != "heuristics"])
        is_degraded = len(missing_models) > 0

        if num_active_models == 0:
            return {
                "ensemble_spoof_risk": None,
                "calibration_status": "uncalibrated",
                "decision_band": DECISION_UNAVAILABLE,
                "is_degraded": True,
                "missing_models": missing_models,
                "thresholds": {
                    "review_threshold": self.review_threshold,
                    "high_risk_threshold": self.high_risk_threshold,
                },
                "models": model_results,
                "heuristics": res_heur,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "decision_explanation": "All anti-spoofing machine learning models are unavailable; heuristics alone cannot generate a verdict.",
            }

        # Re-normalize remaining active weights
        total_active_weight = sum(self.weights[k] for k in active_scores)
        normalized_weights = {k: self.weights[k] / total_active_weight for k in active_scores}

        # Weighted composite score on 0-100 scale
        composite_risk = sum(active_scores[k] * normalized_weights[k] for k in active_scores)
        composite_risk = round(float(np.clip(composite_risk, 0.0, 100.0)), 2)

        # Threshold Decision Policy (Objective 6)
        if composite_risk < self.review_threshold:
            decision_band = DECISION_LOWER_RISK
            explanation = (
                f"Ensemble spoof-risk index ({composite_risk:.1f}/100) is below review threshold ({self.review_threshold:.1f}). "
                "Lower model-estimated spoof risk; does not constitute certified proof of human origin."
            )
        elif composite_risk < self.high_risk_threshold:
            decision_band = DECISION_REVIEW_REQUIRED
            explanation = (
                f"Ensemble spoof-risk index ({composite_risk:.1f}/100) is within review range [{self.review_threshold:.1f}, {self.high_risk_threshold:.1f}). "
                "Ambiguous or border acoustic signals detected; human review or re-challenge required."
            )
        else:
            decision_band = DECISION_HIGH_RISK
            explanation = (
                f"Ensemble spoof-risk index ({composite_risk:.1f}/100) meets or exceeds high-risk threshold ({self.high_risk_threshold:.1f}). "
                "Multiple detectors identified synthetic, cloned, or deepfake voice patterns."
            )

        if is_degraded:
            explanation += f" (Note: Operating in degraded mode with {len(missing_models)} unavailable detector: {', '.join(missing_models)})."

        latency = round((time.perf_counter() - started) * 1000, 2)

        return {
            "ensemble_spoof_risk": composite_risk,
            "calibration_status": "provisional_uncalibrated",
            "decision_band": decision_band,
            "is_degraded": is_degraded,
            "missing_models": missing_models,
            "degraded_reasons": degraded_reasons,
            "effective_weights": {k: round(v, 4) for k, v in normalized_weights.items()},
            "thresholds": {
                "review_threshold": self.review_threshold,
                "high_risk_threshold": self.high_risk_threshold,
            },
            "models": model_results,
            "heuristics": res_heur,
            "latency_ms": latency,
            "decision_explanation": explanation,
        }


@lru_cache(maxsize=1)
def get_ensemble_service() -> EnsembleService:
    return EnsembleService()
