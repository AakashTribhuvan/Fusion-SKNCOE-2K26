import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import soundfile as sf

from app.services.synchronization_service import SynchronizationService
from app.services.aasist_detector import AASISTDetector
from app.services.spoof_detector_service import SpoofDetectorService, TrainedBaselineDetector
from app.services.result_service import ResultService
from app.core.config import settings
from app.training.baseline import LogisticBaseline
from app.training.dataset import AudioExample, grouped_split, load_manifest
from app.training.metrics import calibrate_eer_threshold, classification_metrics
from app.training.evaluate import evaluate
from app.training.features import FEATURE_NAMES
from app.training.train import train


def test_result_service_requires_complete_audio_phrase_and_spoof_checks():
    inputs = {
        "session_id": "test-session",
        "audio_quality": {"status": "pass"},
        "speech_detection": {"status": "completed", "speech_detected": True},
        "phrase_verification": {"status": "completed", "exact_match": True},
        "feature_extraction": {},
        "spoof_detection": {"status": "completed", "predicted_class": "genuine"},
        "audio_video_sync": {},
        "started_at": perf_counter(),
    }
    assert ResultService.build(**inputs)["overall_status"] == "pass"

    phrase_mismatch = {
        **inputs,
        "phrase_verification": {"status": "fail", "exact_match": False},
    }
    assert ResultService.build(**phrase_mismatch)["overall_status"] == "fail"

    spoof_detected = {
        **inputs,
        "spoof_detection": {"status": "completed", "predicted_class": "spoof"},
    }
    assert ResultService.build(**spoof_detected)["overall_status"] == "fail"

    unusable_audio = {
        **inputs,
        "audio_quality": {"status": "fail"},
        "speech_detection": {"status": "fail", "speech_detected": False},
        "phrase_verification": {"status": "incomplete", "exact_match": None},
        "spoof_detection": {"status": "incomplete", "predicted_class": None},
    }
    assert ResultService.build(**unusable_audio)["overall_status"] == "incomplete"


def test_grouped_split_has_no_group_leakage_and_keeps_both_classes():
    examples = [
        AudioExample(path=Path("unused.wav"), label=label, group_id=f"source-{group}")
        for group in range(6)
        for label in (0, 1)
    ]
    splits = grouped_split(examples, seed=3)
    groups = {key: {example.group_id for example in value} for key, value in splits.items()}

    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    for split in splits.values():
        assert {example.label for example in split} == {0, 1}


def test_grouped_split_rejects_too_few_groups():
    examples = [
        AudioExample(path=Path("unused.wav"), label=label, group_id=f"source-{group}")
        for group in range(2)
        for label in (0, 1)
    ]
    with pytest.raises(ValueError, match="three distinct"):
        grouped_split(examples)


def test_manifest_rejects_paths_outside_audio_root(tmp_path):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio_path", "label", "group_id"])
        writer.writeheader()
        writer.writerow({"audio_path": "../outside.wav", "label": "genuine", "group_id": "speaker-1"})
        writer.writerow({"audio_path": "other.wav", "label": "spoof", "group_id": "speaker-2"})
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(manifest, audio_root)


def test_metrics_calibrate_eer_and_report_far_frr():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    calibration = calibrate_eer_threshold(labels, scores)
    metrics = classification_metrics(labels, scores, calibration["threshold"])

    assert calibration["eer"] == 0
    assert metrics["false_accept_rate"] == 0
    assert metrics["false_reject_rate"] == 0


def test_logistic_baseline_checkpoint_round_trip(tmp_path):
    features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float64)
    features[1] = 0.2
    features[2] = 0.8
    features[3] = 1.0
    labels = np.asarray([0, 0, 1, 1])
    model = LogisticBaseline.fit(features, labels, epochs=300)
    checkpoint = tmp_path / "baseline.json"
    model.save(checkpoint, {"model_version": "test.1", "sample_rate": 16000})
    loaded, payload = LogisticBaseline.from_file(checkpoint)

    assert payload["model_version"] == "test.1"
    np.testing.assert_allclose(model.predict_spoof_scores(features), loaded.predict_spoof_scores(features))


def test_disabled_spoof_detector_stays_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "spoof_detector_enabled", False)
    result = SpoofDetectorService().run(np.zeros(16000, dtype=np.float32), 16000)
    assert result["status"] == "unavailable"
    assert result["predicted_class"] in {None, "MODEL_UNAVAILABLE"}
    assert result["genuine_score"] is None
    assert result["spoof_score"] is None


def test_pretrained_aasist_checkpoint_runs_and_reports_uncalibrated_scores():
    checkpoint = Path(__file__).resolve().parents[1] / "app" / "models" / "weights" / "AASIST.pth"
    detector = AASISTDetector(checkpoint)
    audio = (0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000, endpoint=False))).astype(np.float32)
    result = detector.predict(audio, 16000)

    assert result["status"] == "completed"
    assert result["model_name"] == "AASIST"
    assert result["model_version"]
    assert result["predicted_class"] in {"genuine", "spoof", "MODEL_PREDICTS_BONA_FIDE", "MODEL_PREDICTS_SPOOF"}
    assert 0.0 <= result["genuine_score"] <= 1.0
    assert 0.0 <= result["spoof_score"] <= 1.0
    assert result["genuine_score"] + result["spoof_score"] == pytest.approx(1.0)
    assert "not calibrated probabilities" in result["score_semantics"]


def test_sync_missing_evidence_is_explicit_and_supplied_evidence_is_analyzed():
    unavailable = SynchronizationService.analyze()
    assert unavailable["status"] == "unavailable"
    assert unavailable["timing_offset_seconds"] is None

    frames = [
        {"timestamp_seconds": index / 10, "mouth_motion": 1.0 if 0.4 <= index / 10 <= 0.8 else 0.0}
        for index in range(15)
    ]
    result = SynchronizationService.analyze(
        audio_segments=[{"start": 0.4, "end": 0.8}],
        video_frames=frames,
    )
    assert result["status"] == "completed"
    assert result["evidence_quality"] == "experimental"
    assert "not proof" in result["reason"]


def test_training_pipeline_outputs_versioned_checkpoint_metrics_and_registry(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio_path", "label", "group_id", "attack_type"])
        writer.writeheader()
        for group in range(6):
            for label, name in ((0, "genuine"), (1, "spoof")):
                filename = f"group-{group}-{name}.wav"
                frequency = 180 + group * 7 + (label * 100)
                time_axis = np.linspace(0, 1, 16000, endpoint=False)
                signal = (0.3 * np.sin(2 * np.pi * frequency * time_axis)).astype(np.float32)
                sf.write(tmp_path / filename, signal, 16000)
                writer.writerow({
                    "audio_path": filename,
                    "label": name,
                    "group_id": f"speaker-{group}",
                    "attack_type": "genuine-source" if label == 0 else f"training-attack-{group % 2}",
                })

    output = tmp_path / "artifacts" / "checkpoint.json"
    trained = train(manifest_path, None, output, sample_rate=16000, seed=9)

    assert output.is_file()
    assert (output.parent / "model-registry.json").is_file()
    assert trained["metadata"]["held_out_test"]["false_accept_rate"] is not None
    assert trained["metadata"]["held_out_test"]["false_reject_rate"] is not None
    assert "warning" in trained["metadata"]
    assert trained["metadata"]["held_out_test"]["equal_error_rate"] is not None

    detector = TrainedBaselineDetector(output)
    prediction = detector.predict(
        (0.3 * np.sin(2 * np.pi * 200 * np.linspace(0, 1, 16000, endpoint=False))).astype(np.float32),
        16000,
    )
    assert prediction["status"] == "completed"
    assert prediction["predicted_class"] in {"genuine", "spoof", "MODEL_PREDICTS_BONA_FIDE", "MODEL_PREDICTS_SPOOF"}
    assert 0.0 <= prediction["spoof_score"] <= 1.0

    heldout_manifest = tmp_path / "heldout.csv"
    with heldout_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio_path", "label", "group_id", "attack_type"])
        writer.writeheader()
        for group in ("heldout-a", "heldout-b"):
            for label, name in ((0, "genuine"), (1, "spoof")):
                filename = f"{group}-{name}.wav"
                frequency = (220 if label == 0 else 420) + (0 if group == "heldout-a" else 30)
                time_axis = np.linspace(0, 1, 16000, endpoint=False)
                signal = (0.3 * np.sin(2 * np.pi * frequency * time_axis)).astype(np.float32)
                sf.write(tmp_path / filename, signal, 16000)
                writer.writerow({
                    "audio_path": filename,
                    "label": name,
                    "group_id": group,
                    "attack_type": "genuine-source" if label == 0 else "novel-neural-tts",
                })
    evaluation = evaluate(output, heldout_manifest, unseen_attack_types={"novel-neural-tts"})
    assert evaluation["equal_error_rate"] is not None
    assert evaluation["metrics"]["false_accept_rate"] is not None
    assert evaluation["metrics"]["false_reject_rate"] is not None
    assert evaluation["unseen_attack_types"] == ["novel-neural-tts"]
    assert "novel-neural-tts" in evaluation["attack_type_metrics"]
    with pytest.raises(ValueError, match="used during training or calibration"):
        evaluate(output, manifest_path)
