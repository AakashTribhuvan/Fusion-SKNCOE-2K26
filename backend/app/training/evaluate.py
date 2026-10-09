from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.training.baseline import LogisticBaseline
from app.training.dataset import load_example_features, load_manifest
from app.training.metrics import calibrate_eer_threshold, classification_metrics


def evaluate(
    checkpoint: Path,
    manifest: Path,
    audio_root: Path | None = None,
    calibrate: bool = False,
    unseen_attack_types: set[str] | None = None,
) -> dict:
    model, metadata = LogisticBaseline.from_file(checkpoint)
    examples = load_manifest(manifest, audio_root)
    trained_groups = {
        group_id
        for split_groups in metadata["training_metadata"]["split_groups"].values()
        for group_id in split_groups
    }
    if any(example.group_id in trained_groups for example in examples):
        raise ValueError("Evaluation manifest contains a group used during training or calibration; provide an independent evaluation set.")
    training_attack_types = set(
        metadata["training_metadata"].get("spoof_attack_types_by_split", {}).get("train", [])
    )
    evaluation_spoof_attack_types = {example.attack_type for example in examples if example.label == 1}
    if unseen_attack_types is not None:
        if not unseen_attack_types:
            raise ValueError("At least one unseen attack type must be specified.")
        if not evaluation_spoof_attack_types.issubset(unseen_attack_types):
            raise ValueError("Evaluation manifest includes spoof attack types not listed as unseen.")
        if evaluation_spoof_attack_types & training_attack_types:
            raise ValueError("An attack type marked unseen was present in the training split.")
    feature_rows = [load_example_features(example, metadata["sample_rate"]) for example in examples]
    features = np.vstack(feature_rows)
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    started = time.perf_counter()
    scores = model.predict_spoof_scores(features)
    elapsed_ms = (time.perf_counter() - started) * 1000
    calibration = calibrate_eer_threshold(labels, scores) if calibrate else None
    eer_report = calibrate_eer_threshold(labels, scores)
    threshold = calibration["threshold"] if calibration else model.threshold
    attack_type_metrics = {}
    for attack_type in sorted({example.attack_type for example in examples}):
        mask = np.asarray([example.attack_type == attack_type for example in examples])
        attack_type_metrics[attack_type] = classification_metrics(labels[mask], scores[mask], threshold)
    return {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "checkpoint": checkpoint.name,
        "sample_rate": metadata["sample_rate"],
        "evaluation_examples": len(examples),
        "evaluation_groups": len({example.group_id for example in examples}),
        "unseen_attack_types": sorted(evaluation_spoof_attack_types) if unseen_attack_types is not None else [],
        "attack_type_metrics": attack_type_metrics,
        "threshold": threshold,
        "metrics": classification_metrics(labels, scores, threshold),
        "equal_error_rate": eer_report["eer"],
        "eer_threshold_for_reporting_only": eer_report["threshold"],
        "equal_error_rate_calibration": calibration,
        "mean_inference_latency_ms_per_clip": elapsed_ms / max(1, len(features)),
        "warning": "Evaluation results only characterize this supplied evaluation manifest.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FUSION spoof baseline on an independent manifest.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--calibrate", action="store_true", help="Calibrate threshold on this set; do not use for final test reporting.")
    parser.add_argument(
        "--unseen-attack-types",
        nargs="*",
        help="Declare spoof attack types that must be absent from the training split.",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(
        args.checkpoint,
        args.manifest,
        args.audio_root,
        args.calibrate,
        set(args.unseen_attack_types) if args.unseen_attack_types is not None else None,
    ), indent=2))


if __name__ == "__main__":
    main()
