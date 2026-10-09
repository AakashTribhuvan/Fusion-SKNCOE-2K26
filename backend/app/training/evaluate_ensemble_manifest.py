from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.services.ensemble_service import EnsembleService, get_ensemble_service
from app.services.shared_audio_preprocessor import SharedAudioPreprocessor


def calculate_metrics(y_true: list[int], y_pred: list[int], scores: list[float]) -> dict[str, Any]:
    """Calculate confusion matrix, FAR, FRR, Precision, Recall, and EER.
    
    Class encoding: 0 = BONAFIDE (genuine human), 1 = SPOOF (synthetic / deepfake).
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)  # genuine falsely rejected as spoof
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)  # spoof falsely accepted as genuine
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    total_spoof = tp + fn
    total_bona_fide = tn + fp

    # Security terminology:
    # False Acceptance Rate (FAR): Spoof accepted as genuine = FN / total_spoof
    far = (fn / total_spoof) if total_spoof > 0 else 0.0
    # False Rejection Rate (FRR): Genuine rejected as spoof = FP / total_bona_fide
    frr = (fp / total_bona_fide) if total_bona_fide > 0 else 0.0

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / total_spoof) if total_spoof > 0 else 0.0
    accuracy = ((tp + tn) / len(y_true)) if y_true else 0.0

    # Equal Error Rate (EER) calculation
    eer = None
    eer_threshold = None
    if total_spoof > 0 and total_bona_fide > 0 and len(scores) == len(y_true):
        # Sort thresholds by score
        thresholds = sorted(set(scores))
        min_diff = float("inf")
        for th in thresholds:
            cur_fn = sum(1 for yt, s in zip(y_true, scores) if yt == 1 and s < th)
            cur_fp = sum(1 for yt, s in zip(y_true, scores) if yt == 0 and s >= th)
            cur_far = cur_fn / total_spoof
            cur_frr = cur_fp / total_bona_fide
            diff = abs(cur_far - cur_frr)
            if diff < min_diff:
                min_diff = diff
                eer = (cur_far + cur_frr) / 2.0
                eer_threshold = th

    return {
        "confusion_matrix": {
            "true_spoof_detected (TP)": tp,
            "false_alarm_genuine_rejected (FP)": fp,
            "missed_spoof_accepted (FN)": fn,
            "true_genuine_accepted (TN)": tn,
        },
        "false_acceptance_rate_far": round(far, 4),
        "false_rejection_rate_frr": round(frr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "eer": round(eer, 4) if eer is not None else None,
        "eer_threshold": round(eer_threshold, 4) if eer_threshold is not None else None,
    }


def evaluate_manifest(manifest_csv: Path, output_json: Path | None = None) -> dict[str, Any]:
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    ensemble = get_ensemble_service()

    records = []
    with open(manifest_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        raise ValueError("Manifest is empty.")

    results_per_audio: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred_ensemble: list[int] = []
    ensemble_scores: list[float] = []

    model_b_preds: list[int] = []
    model_b_scores: list[float] = []
    model_c_preds: list[int] = []
    model_c_scores: list[float] = []

    source_breakdown: dict[str, dict[str, int]] = {}

    for row in records:
        audio_path_str = row.get("file_path") or row.get("path")
        label_str = (row.get("label") or row.get("ground_truth") or "").strip().upper()
        source_type = (row.get("source_type") or row.get("source") or "unknown").strip()

        audio_path = Path(audio_path_str)
        if not audio_path.is_file():
            results_per_audio.append({
                "file_path": str(audio_path),
                "error": "FILE_NOT_FOUND",
                "label": label_str,
            })
            continue

        true_cls = 1 if label_str in ("SPOOF", "FAKE", "SYNTHETIC") else 0

        t0 = time.perf_counter()
        try:
            with open(audio_path, "rb") as af:
                audio, sr = SharedAudioPreprocessor.decode_and_validate(af.read())
            eval_res = ensemble.evaluate(audio, sr)
            latency = round((time.perf_counter() - t0) * 1000, 2)

            comp_risk = eval_res.get("ensemble_spoof_risk", 50.0)
            decision = eval_res.get("decision_band")

            # Pred: 1 if high risk (>= 75.0) or review (>= 70.0), else 0
            pred_cls = 1 if comp_risk >= ensemble.review_threshold else 0

            y_true.append(true_cls)
            y_pred_ensemble.append(pred_cls)
            ensemble_scores.append(comp_risk)

            mb_score = eval_res.get("models", {}).get("model_b", {}).get("spoof_score")
            if mb_score is not None:
                model_b_scores.append(mb_score)
                model_b_preds.append(1 if mb_score >= 50.0 else 0)

            mc_score = eval_res.get("models", {}).get("model_c", {}).get("spoof_score")
            if mc_score is not None:
                model_c_scores.append(mc_score)
                model_c_preds.append(1 if mc_score >= 50.0 else 0)

            # Track source breakdown
            if source_type not in source_breakdown:
                source_breakdown[source_type] = {"total": 0, "detected_spoof": 0, "detected_genuine": 0}
            source_breakdown[source_type]["total"] += 1
            if pred_cls == 1:
                source_breakdown[source_type]["detected_spoof"] += 1
            else:
                source_breakdown[source_type]["detected_genuine"] += 1

            results_per_audio.append({
                "file_path": str(audio_path),
                "ground_truth": label_str,
                "source_type": source_type,
                "ensemble_spoof_risk": comp_risk,
                "decision_band": decision,
                "predicted_class": pred_cls,
                "latency_ms": latency,
                "is_degraded": eval_res.get("is_degraded"),
                "models": {
                    "model_a_w2v2_aasist": eval_res.get("models", {}).get("model_a", {}).get("status"),
                    "model_b_aasist_score": mb_score,
                    "model_c_wav2vec2_score": mc_score,
                },
                "heuristics_score": eval_res.get("heuristics", {}).get("heuristic_risk_score"),
            })

        except Exception as exc:
            results_per_audio.append({
                "file_path": str(audio_path),
                "error": str(exc),
                "ground_truth": label_str,
            })

    report = {
        "manifest_path": str(manifest_csv),
        "total_evaluated": len(y_true),
        "total_failures": len(records) - len(y_true),
        "ensemble_metrics": calculate_metrics(y_true, y_pred_ensemble, ensemble_scores),
        "model_b_aasist_metrics": calculate_metrics(y_true, model_b_preds, model_b_scores) if model_b_preds else None,
        "model_c_wav2vec2_metrics": calculate_metrics(y_true, model_c_preds, model_c_scores) if model_c_preds else None,
        "source_breakdown": source_breakdown,
        "sample_results": results_per_audio,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as out:
            json.dump(report, out, indent=2)
        print(f"Report saved to {output_json}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multi-model ensemble on audio manifest CSV.")
    parser.add_argument("manifest", type=Path, help="Path to manifest CSV")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    res = evaluate_manifest(args.manifest, args.output)
    print(json.dumps(res["ensemble_metrics"], indent=2))
