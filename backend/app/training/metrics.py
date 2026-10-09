from __future__ import annotations

import numpy as np


def classification_metrics(labels: np.ndarray, spoof_scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    spoof_scores = np.asarray(spoof_scores, dtype=np.float64)
    predicted_spoof = spoof_scores >= threshold
    true_spoof = labels == 1
    true_genuine = labels == 0

    false_accepts = int(np.sum(true_spoof & ~predicted_spoof))
    false_rejects = int(np.sum(true_genuine & predicted_spoof))
    spoof_count = int(np.sum(true_spoof))
    genuine_count = int(np.sum(true_genuine))
    far = false_accepts / spoof_count if spoof_count else None
    frr = false_rejects / genuine_count if genuine_count else None
    accuracy = float(np.mean(predicted_spoof == true_spoof)) if labels.size else None
    return {
        "threshold": float(threshold),
        "false_accept_rate": far,
        "false_reject_rate": frr,
        "accuracy": accuracy,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "spoof_count": spoof_count,
        "genuine_count": genuine_count,
    }


def calibrate_eer_threshold(labels: np.ndarray, spoof_scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    spoof_scores = np.asarray(spoof_scores, dtype=np.float64)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("EER calibration requires both genuine and spoof examples.")
    if not np.isfinite(spoof_scores).all():
        raise ValueError("Scores must be finite.")

    candidates = np.unique(np.concatenate(([0.0], spoof_scores, [1.0])))
    results = [classification_metrics(labels, spoof_scores, float(threshold)) for threshold in candidates]
    selected = min(
        results,
        key=lambda item: (
            abs(item["false_accept_rate"] - item["false_reject_rate"]),
            item["false_accept_rate"] + item["false_reject_rate"],
        ),
    )
    eer = (selected["false_accept_rate"] + selected["false_reject_rate"]) / 2
    return {"threshold": selected["threshold"], "eer": float(eer), **selected}
