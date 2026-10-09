from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.services.aasist_detector import (
    AASISTDetector,
    STATUS_BONA_FIDE,
    STATUS_SPOOF,
    STATUS_INDETERMINATE,
    STATUS_UNAVAILABLE,
    STATUS_ERROR,
)
from app.services.audio_decode_service import AudioDecodeService
from app.training.metrics import calibrate_eer_threshold

LABEL_MAP = {
    "bonafide": "BONAFIDE",
    "bona_fide": "BONAFIDE",
    "genuine": "BONAFIDE",
    "real": "BONAFIDE",
    "human": "BONAFIDE",
    "spoof": "SPOOF",
    "fake": "SPOOF",
    "ai": "SPOOF",
    "tts": "SPOOF",
    "cloned": "SPOOF",
}


def evaluate_manifest(
    manifest_path: Path,
    audio_root: Path | None = None,
    threshold: float = 0.0,
    detector: Any | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    root = (Path(audio_root) if audio_root else manifest_path.parent).resolve()

    if detector is None:
        if not settings.aasist_model_path.is_file():
            raise FileNotFoundError(f"AASIST checkpoint not found at: {settings.aasist_model_path}")
        detector = AASISTDetector(settings.aasist_model_path, threshold=threshold)

    results_table = []
    tp = fp = tn = fn = 0
    inference_failures = 0
    latencies = []
    source_stats: dict[str, dict[str, Any]] = {}

    y_true_binary = []
    y_scores_llr = []

    with manifest_path.open("r", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row_idx, row in enumerate(reader, start=2):
            raw_path = row.get("audio_path") or row.get("file_path") or row.get("path")
            raw_label = (row.get("label") or "").strip().lower()
            source_type = (row.get("source_type") or "unspecified").strip().lower()
            speaker_id = (row.get("speaker_id") or "unknown").strip()

            if not raw_path or raw_label not in LABEL_MAP:
                continue

            ground_truth = LABEL_MAP[raw_label]
            full_audio_path = (root / raw_path).resolve()

            if not full_audio_path.is_file():
                inference_failures += 1
                results_table.append({
                    "row": row_idx,
                    "file": raw_path,
                    "ground_truth": ground_truth,
                    "source_type": source_type,
                    "status": "FILE_NOT_FOUND",
                    "raw_score": None,
                    "predicted_class": STATUS_ERROR,
                    "latency_ms": 0.0,
                })
                continue

            # Load and decode audio
            try:
                audio_bytes = full_audio_path.read_bytes()
                audio_np, sample_rate = AudioDecodeService.decode_bytes(audio_bytes)
                pred = detector.predict(audio_np, sample_rate)
            except Exception as exc:
                inference_failures += 1
                results_table.append({
                    "row": row_idx,
                    "file": raw_path,
                    "ground_truth": ground_truth,
                    "source_type": source_type,
                    "status": "PROCESSING_ERROR",
                    "raw_score": None,
                    "predicted_class": STATUS_ERROR,
                    "latency_ms": 0.0,
                    "error": str(exc),
                })
                continue

            predicted_class = pred.get("predicted_class")
            raw_score = pred.get("raw_score")
            latency = pred.get("inference_latency_ms", 0.0)
            latencies.append(latency)

            # Confusion matrix mapping: Positive = BONAFIDE, Negative = SPOOF
            is_pred_bonafide = predicted_class == STATUS_BONA_FIDE
            is_pred_spoof = predicted_class == STATUS_SPOOF

            if ground_truth == "BONAFIDE":
                y_true_binary.append(1)
                if raw_score is not None:
                    y_scores_llr.append(raw_score)
                if is_pred_bonafide:
                    tp += 1
                elif is_pred_spoof:
                    fn += 1
            else:  # SPOOF
                y_true_binary.append(0)
                if raw_score is not None:
                    y_scores_llr.append(raw_score)
                if is_pred_spoof:
                    tn += 1
                elif is_pred_bonafide:
                    fp += 1

            # Grouped stats by audio source
            if source_type not in source_stats:
                source_stats[source_type] = {"total": 0, "correct": 0, "false_accepts": 0, "false_rejects": 0}
            source_stats[source_type]["total"] += 1
            if (ground_truth == "BONAFIDE" and is_pred_bonafide) or (ground_truth == "SPOOF" and is_pred_spoof):
                source_stats[source_type]["correct"] += 1
            if ground_truth == "SPOOF" and is_pred_bonafide:
                source_stats[source_type]["false_accepts"] += 1
            if ground_truth == "BONAFIDE" and is_pred_spoof:
                source_stats[source_type]["false_rejects"] += 1

            results_table.append({
                "row": row_idx,
                "file": raw_path,
                "ground_truth": ground_truth,
                "source_type": source_type,
                "speaker_id": speaker_id,
                "raw_score": raw_score,
                "predicted_class": predicted_class,
                "latency_ms": latency,
                "status": pred.get("status"),
            })

    total_evaluated = tp + fp + tn + fn
    # False Acceptance Rate: Spoofs incorrectly accepted as genuine
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # False Rejection Rate: Bona fide incorrectly rejected as spoof
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / total_evaluated if total_evaluated > 0 else 0.0

    # Calculate EER if both classes are present
    eer = None
    if len(set(y_true_binary)) > 1 and len(y_scores_llr) == len(y_true_binary):
        try:
            # calibrate_eer_threshold expects label=1 for spoof, 0 for genuine
            spoof_labels = np.asarray([1 - y for y in y_true_binary], dtype=np.int64)
            # Higher spoof score = -raw_score (since raw_score is LLR bona fide - spoof)
            spoof_probs = 1.0 / (1.0 + np.exp(np.clip(y_scores_llr, -30.0, 30.0)))
            eer_res = calibrate_eer_threshold(spoof_labels, spoof_probs)
            eer = float(eer_res["eer"])
        except Exception:
            eer = None

    return {
        "manifest_path": str(manifest_path),
        "total_rows": len(results_table),
        "total_evaluated": total_evaluated,
        "inference_failures": inference_failures,
        "confusion_matrix": {
            "true_positive_bonafide": tp,
            "true_negative_spoof": tn,
            "false_positive_false_acceptance": fp,
            "false_negative_false_rejection": fn,
        },
        "metrics": {
            "false_acceptance_rate_far": round(far, 4),
            "false_rejection_rate_frr": round(frr, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "equal_error_rate_eer": round(eer, 4) if eer is not None else "N/A (requires multi-class dataset)",
            "average_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        },
        "source_breakdown": source_stats,
        "sample_results": results_table,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate anti-spoofing detector on a labelled CSV manifest.")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to CSV manifest")
    parser.add_argument("--audio-root", default=None, type=Path, help="Base directory for audio paths")
    parser.add_argument("--threshold", default=0.0, type=float, help="Decision threshold for LLR")
    parser.add_argument("--output", default=None, type=Path, help="Output JSON results path")
    args = parser.parse_args()

    results = evaluate_manifest(args.manifest, args.audio_root, args.threshold)
    print(json.dumps({k: v for k, v in results.items() if k != "sample_results"}, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
