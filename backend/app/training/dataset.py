from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from app.training.features import extract_feature_vector

LABELS = {
    "genuine": 0,
    "bona_fide": 0,
    "bonafide": 0,
    "real": 0,
    "spoof": 1,
    "fake": 1,
}
MANIFEST_COLUMNS = {"audio_path", "label", "group_id"}


@dataclass(frozen=True)
class AudioExample:
    path: Path
    label: int
    group_id: str
    attack_type: str = "unspecified"


def load_manifest(manifest_path: Path, audio_root: Path | None = None) -> list[AudioExample]:
    manifest_path = manifest_path.resolve()
    root = (audio_root or manifest_path.parent).resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if not reader.fieldnames or not MANIFEST_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("Manifest must include audio_path,label,group_id columns.")

        examples: list[AudioExample] = []
        for line_number, row in enumerate(reader, start=2):
            label_text = (row.get("label") or "").strip().lower()
            group_id = (row.get("group_id") or "").strip()
            relative_path = (row.get("audio_path") or "").strip()
            if label_text not in LABELS or not group_id or not relative_path:
                raise ValueError(f"Invalid label, group_id, or audio_path at manifest line {line_number}.")
            path = (root / relative_path).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Audio path escapes the configured audio root at line {line_number}.")
            if not path.is_file():
                raise ValueError(f"Audio file does not exist at manifest line {line_number}.")
            examples.append(
                AudioExample(
                    path=path,
                    label=LABELS[label_text],
                    group_id=group_id,
                    attack_type=(row.get("attack_type") or "unspecified").strip() or "unspecified",
                )
            )

    if not examples:
        raise ValueError("Manifest contains no audio examples.")
    labels = {example.label for example in examples}
    if labels != {0, 1}:
        raise ValueError("Manifest must contain both genuine/bona_fide and spoof examples.")
    return examples


def load_example_features(example: AudioExample, target_sample_rate: int) -> np.ndarray:
    try:
        audio, sample_rate = sf.read(example.path, dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Unable to decode audio file {example.path.name}.") from exc
    if audio.size == 0:
        raise ValueError(f"Audio file {example.path.name} has no samples.")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
    return extract_feature_vector(np.asarray(audio, dtype=np.float32), target_sample_rate)


def grouped_split(
    examples: list[AudioExample],
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, list[AudioExample]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction plus validation_fraction must be less than one.")

    groups = sorted({example.group_id for example in examples})
    if len(groups) < 3:
        raise ValueError("At least three distinct group_id values are required for train/validation/test splitting.")

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    train_end = min(max(1, int(round(len(groups) * train_fraction))), len(groups) - 2)
    validation_end = min(
        max(train_end + 1, int(round(len(groups) * (train_fraction + validation_fraction)))),
        len(groups) - 1,
    )
    group_assignments = {
        group_id: split
        for split, group_list in (
            ("train", groups[:train_end]),
            ("validation", groups[train_end:validation_end]),
            ("test", groups[validation_end:]),
        )
        for group_id in group_list
    }

    splits = {
        split: [example for example in examples if group_assignments[example.group_id] == split]
        for split in ("train", "validation", "test")
    }
    split_groups = {name: {example.group_id for example in split_examples} for name, split_examples in splits.items()}
    if split_groups["train"] & split_groups["validation"] or split_groups["train"] & split_groups["test"] or split_groups["validation"] & split_groups["test"]:
        raise RuntimeError("Internal error: group leakage detected.")
    for name, split_examples in splits.items():
        if {example.label for example in split_examples} != {0, 1}:
            raise ValueError(f"The {name} split must contain both classes; add more groups or adjust the dataset.")
    return splits
