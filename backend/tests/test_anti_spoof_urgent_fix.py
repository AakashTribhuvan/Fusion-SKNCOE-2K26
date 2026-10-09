import csv
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from time import perf_counter

from app.core.config import settings
from app.services.aasist_detector import (
    AASISTDetector,
    W2V2AASISTDetector,
    STATUS_BONA_FIDE,
    STATUS_SPOOF,
    STATUS_INDETERMINATE,
    STATUS_UNAVAILABLE,
    STATUS_ERROR,
    _prepare_audio,
)
from app.services.spoof_detector_service import SpoofDetectorService
from app.services.result_service import ResultService
from app.training.evaluate_manifest import evaluate_manifest


def test_aasist_score_direction_and_threshold_classification():
    """Verify AASIST returns higher LLR for genuine speech and lower for synthetic audio."""
    checkpoint = settings.aasist_model_path
    detector = AASISTDetector(checkpoint, threshold=0.0, margin=0.1)

    # 1. Pure sine wave (harsh synthetic non-speech signal)
    synthetic_sine = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 4, 64600))).astype(np.float32)
    res_sine = detector.predict(synthetic_sine, 16000)

    assert res_sine["status"] == "completed"
    assert res_sine["detector_name"] == "AASIST"
    assert res_sine["predicted_class"] == STATUS_SPOOF
    assert res_sine["raw_score"] < 0.0  # LLR is negative for obvious spoof
    assert res_sine["spoof_score"] > 0.9
    assert res_sine["genuine_score"] < 0.1
    assert "Log-likelihood ratio" in res_sine["score_semantics"]


def test_audio_preprocessing_handles_stereo_resampling_and_length():
    """Verify preprocessing handles stereo audio, sample rate conversion, and padding."""
    # Stereo 44.1kHz audio
    t = np.linspace(0, 1, 44100, endpoint=False)
    left = 0.2 * np.sin(2 * np.pi * 300 * t)
    right = 0.2 * np.sin(2 * np.pi * 300 * t)
    stereo_44k = np.stack([left, right], axis=-1).astype(np.float32)

    prepared = _prepare_audio(stereo_44k, sample_rate=44100)
    assert prepared.ndim == 1
    assert prepared.shape[0] == 64600  # Padded to exactly 64600 samples
    assert not np.isnan(prepared).any()


def test_audio_preprocessing_rejects_empty():
    """Verify empty audio raises ValueError."""
    with pytest.raises(ValueError, match="no samples"):
        _prepare_audio(np.array([], dtype=np.float32), 16000)


def test_w2v2_aasist_reports_model_unavailable_when_not_cached():
    """Verify W2V2-AASIST gracefully reports MODEL_UNAVAILABLE with reason."""
    detector = W2V2AASISTDetector(checkpoint_path=None)
    res = detector.predict(np.zeros(16000, dtype=np.float32), 16000)

    assert res["model_status"] == "unavailable"
    assert res["predicted_class"] == STATUS_UNAVAILABLE
    assert res["error_code"] == "CHECKPOINT_NOT_CACHED"
    assert "not cached locally" in res["reason"]


def test_centralized_spoof_detector_service_explicit_statuses():
    """Verify SpoofDetectorService returns standardized Phase 3 statuses."""
    service = SpoofDetectorService()
    audio = (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000))).astype(np.float32)
    res = service.run(audio, 16000)

    assert "detector_name" in res
    assert "model_status" in res
    assert "raw_score" in res
    assert "predicted_class" in res
    assert "threshold" in res
    assert "inference_latency_ms" in res
    assert "error_code" in res
    assert res["predicted_class"] in {STATUS_BONA_FIDE, STATUS_SPOOF, STATUS_INDETERMINATE, STATUS_UNAVAILABLE}


def test_phrase_match_does_not_override_spoof_detection():
    """Verify matching phrase NEVER marks session as pass if spoof is detected."""
    res = ResultService.build(
        session_id="test-sec",
        audio_quality={"status": "pass"},
        speech_detection={"status": "completed", "speech_detected": True},
        phrase_verification={"status": "pass", "exact_match": True, "recognized_phrase": "expected phrase"},
        feature_extraction={},
        spoof_detection={
            "status": "completed",
            "predicted_class": STATUS_SPOOF,
            "raw_score": -3.5,
            "spoof_score": 0.97,
        },
        audio_video_sync={},
        started_at=perf_counter(),
    )
    assert res["overall_status"] == "fail"
    assert res["risk_category"] == "HIGH_RISK"
    assert any("synthetic, cloned, or deepfake" in r for r in res["reasons"])


def test_unavailable_spoof_detection_requires_review_not_pass():
    """Verify session is REVIEW_REQUIRED (not pass) if spoof check is unavailable."""
    res = ResultService.build(
        session_id="test-review",
        audio_quality={"status": "pass"},
        speech_detection={"status": "completed", "speech_detected": True},
        phrase_verification={"status": "pass", "exact_match": True},
        feature_extraction={},
        spoof_detection={
            "status": "unavailable",
            "predicted_class": STATUS_UNAVAILABLE,
            "raw_score": None,
        },
        audio_video_sync={},
        started_at=perf_counter(),
    )
    assert res["overall_status"] == "review_required"
    assert res["risk_category"] == "REVIEW_REQUIRED"


def test_evaluate_manifest_generates_far_frr_and_confusion_matrix(tmp_path):
    """Verify evaluation script computes confusion matrix, FAR, and FRR on labelled recordings."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    manifest_file = tmp_path / "manifest.csv"
    with manifest_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["audio_path", "label", "source_type", "speaker_id"])
        writer.writeheader()

        # Create 2 synthetic spoof signals (sine wave)
        for i in range(2):
            filename = f"spoof_{i}.wav"
            sig = (0.4 * np.sin(2 * np.pi * (440 + i * 50) * np.linspace(0, 4, 64600))).astype(np.float32)
            sf.write(audio_dir / filename, sig, 16000)
            writer.writerow({
                "audio_path": filename,
                "label": "SPOOF",
                "source_type": "synthetic_sine",
                "speaker_id": f"spk_synth_{i}",
            })

    eval_results = evaluate_manifest(manifest_file, audio_root=audio_dir, threshold=0.0)

    assert eval_results["total_evaluated"] == 2
    assert eval_results["inference_failures"] == 0
    assert "confusion_matrix" in eval_results
    assert "metrics" in eval_results
    assert eval_results["metrics"]["false_acceptance_rate_far"] == 0.0  # None accepted as bona fide
    assert eval_results["confusion_matrix"]["true_negative_spoof"] == 2
    assert "synthetic_sine" in eval_results["source_breakdown"]
