from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.training.baseline import LogisticBaseline
from app.training.dataset import grouped_split, load_example_features, load_manifest
from app.training.metrics import calibrate_eer_threshold, classification_metrics


def _matrix(examples, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    feature_rows = [load_example_features(example, sample_rate) for example in examples]
    return np.vstack(feature_rows), np.asarray([example.label for example in examples], dtype=np.int64)


def train(manifest: Path, audio_root: Path | None, output: Path, sample_rate: int = 16000, seed: int = 42) -> dict:
    examples = load_manifest(manifest, audio_root)
    splits = grouped_split(examples, seed=seed)
    train_x, train_y = _matrix(splits["train"], sample_rate)
    validation_x, validation_y = _matrix(splits["validation"], sample_rate)
    test_x, test_y = _matrix(splits["test"], sample_rate)

    model = LogisticBaseline.fit(train_x, train_y)
    validation_scores = model.predict_spoof_scores(validation_x)
    calibration = calibrate_eer_threshold(validation_y, validation_scores)
    model.threshold = calibration["threshold"]

    test_scores = model.predict_spoof_scores(test_x)
    started = time.perf_counter()
    model.predict_spoof_scores(test_x)
    per_clip_latency_ms = (time.perf_counter() - started) * 1000 / max(1, len(test_x))
    test_metrics = classification_metrics(test_y, test_scores, model.threshold)
    test_eer = calibrate_eer_threshold(test_y, test_scores)

    model_version = datetime.now(timezone.utc).strftime("0.1.%Y%m%d%H%M%S")
    metadata = {
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": sample_rate,
        "random_seed": seed,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "dataset_examples": len(examples),
        "dataset_groups": len({example.group_id for example in examples}),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_groups": {name: sorted({row.group_id for row in rows}) for name, rows in splits.items()},
        "spoof_attack_types_by_split": {
            name: sorted({row.attack_type for row in rows if row.label == 1})
            for name, rows in splits.items()
        },
        "calibration": calibration,
        "held_out_test": {
            **test_metrics,
            "equal_error_rate": test_eer["eer"],
            "eer_threshold_for_reporting_only": test_eer["threshold"],
            "mean_inference_latency_ms_per_clip": per_clip_latency_ms,
        },
        "warning": "Research baseline only. Metrics are dataset-specific and are not a production security guarantee.",
    }
    model.save(output, metadata)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    registry_path = output.with_name("model-registry.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"models": []}
    registry["models"].append({
        "model_name": "fusion-mfcc-logistic-baseline",
        "model_version": model_version,
        "checkpoint": output.name,
        "sha256": digest,
        "sample_rate": sample_rate,
        "created_at": metadata["created_at"],
        "metrics": metadata["held_out_test"],
        "calibration": calibration,
    })
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return {"checkpoint": str(output), "registry": str(registry_path), "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the FUSION research baseline.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV containing audio_path,label,group_id.")
    parser.add_argument("--audio-root", type=Path, help="Root for relative audio_path values; defaults to manifest directory.")
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint output JSON path.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train(args.manifest, args.audio_root, args.output, args.sample_rate, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
