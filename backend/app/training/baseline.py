from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.training.features import FEATURE_NAMES


@dataclass
class LogisticBaseline:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float = 0.5

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 0.05,
        epochs: int = 1500,
        l2: float = 0.001,
    ) -> "LogisticBaseline":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.size or features.shape[0] < 2:
            raise ValueError("Training features and labels have invalid dimensions.")
        if set(np.unique(labels).tolist()) != {0.0, 1.0}:
            raise ValueError("Training data must contain both classes.")
        if not np.isfinite(features).all():
            raise ValueError("Training features must be finite.")

        mean = np.mean(features, axis=0)
        scale = np.std(features, axis=0)
        scale[scale < 1e-8] = 1.0
        normalized = (features - mean) / scale
        weights = np.zeros(features.shape[1], dtype=np.float64)
        bias = 0.0

        for _ in range(epochs):
            probabilities = cls._sigmoid(normalized @ weights + bias)
            error = probabilities - labels
            weights -= learning_rate * ((normalized.T @ error) / labels.size + l2 * weights)
            bias -= learning_rate * float(np.mean(error))

        return cls(mean=mean, scale=scale, weights=weights, bias=bias)

    def predict_spoof_scores(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.weights.size:
            raise ValueError("Input feature dimensions do not match this checkpoint.")
        return self._sigmoid(((features - self.mean) / self.scale) @ self.weights + self.bias)

    def to_dict(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_name": "fusion-mfcc-logistic-baseline",
            "model_version": metadata["model_version"],
            "model_type": "research_baseline_logistic_regression",
            "feature_names": list(FEATURE_NAMES),
            "sample_rate": int(metadata["sample_rate"]),
            "spoof_score_semantics": "sigmoid probability from a trained logistic baseline; not calibrated confidence",
            "threshold": float(self.threshold),
            "normalization_mean": self.mean.tolist(),
            "normalization_scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "training_metadata": metadata,
        }

    @classmethod
    def from_file(cls, checkpoint_path: Path) -> tuple["LogisticBaseline", dict[str, Any]]:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("model_type") != "research_baseline_logistic_regression":
            raise ValueError("Unsupported spoof detector checkpoint format.")
        model = cls(
            mean=np.asarray(payload["normalization_mean"], dtype=np.float64),
            scale=np.asarray(payload["normalization_scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            threshold=float(payload.get("threshold", 0.5)),
        )
        if model.mean.shape != model.scale.shape or model.mean.shape != model.weights.shape or model.weights.size != len(FEATURE_NAMES):
            raise ValueError("Checkpoint feature dimensions are invalid.")
        return model, payload

    def save(self, output_path: Path, metadata: dict[str, Any]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(metadata), indent=2), encoding="utf-8")
