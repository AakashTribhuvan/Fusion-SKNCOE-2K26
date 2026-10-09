from __future__ import annotations

from typing import Any

import librosa
import numpy as np


FEATURE_NAMES = (
    *(f"mfcc_{index}_mean" for index in range(13)),
    *(f"mfcc_{index}_std" for index in range(13)),
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "rms_mean",
    "rms_std",
)


def extract_feature_vector(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Expected non-empty mono audio.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13, n_fft=1024, hop_length=256)
    features: list[float] = [
        *np.mean(mfcc, axis=1).tolist(),
        *np.std(mfcc, axis=1).tolist(),
    ]
    for feature in (
        librosa.feature.spectral_centroid(y=audio, sr=sample_rate),
        librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate),
        librosa.feature.spectral_rolloff(y=audio, sr=sample_rate),
        librosa.feature.zero_crossing_rate(audio),
        librosa.feature.rms(y=audio),
    ):
        features.extend((float(np.mean(feature)), float(np.std(feature))))

    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Audio feature extraction returned invalid values.")
    return vector


def extract_with_metadata(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    vector = extract_feature_vector(audio, sample_rate)
    return vector, {
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "sample_rate": sample_rate,
    }
