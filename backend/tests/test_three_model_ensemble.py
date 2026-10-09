from __future__ import annotations

import numpy as np
import pytest
import torch

from app.core.config import settings
from app.services.audio_heuristics import AudioHeuristicsAnalyzer
from app.services.ensemble_service import (
    DECISION_HIGH_RISK,
    DECISION_INDETERMINATE,
    DECISION_LOWER_RISK,
    DECISION_REVIEW_REQUIRED,
    DECISION_UNAVAILABLE,
    EnsembleService,
    ModelA_W2V2AASIST_Adapter,
    ModelB_AASIST_Adapter,
    ModelC_Wav2Vec2Deepfake_Adapter,
    SPOOF_HIGH_RISK_THRESHOLD,
    SPOOF_REVIEW_THRESHOLD,
)
from app.services.result_service import ResultService
from app.services.shared_audio_preprocessor import (
    PreprocessingError,
    SharedAudioPreprocessor,
)


def test_shared_preprocessor_resampling_and_windowing():
    # Generate 1 second of 8kHz sine wave
    t = np.linspace(0, 1, 8000, endpoint=False, dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 1. Resample to 16kHz
    resampled = SharedAudioPreprocessor.resample(sine, orig_sr=8000, target_sr=16000)
    assert len(resampled) == 16000

    # 2. Prepare for AASIST (should tile to exactly 64600 samples)
    prep_aasist = SharedAudioPreprocessor.prepare_for_aasist(sine, sample_rate=8000)
    assert len(prep_aasist) == 64600
    assert prep_aasist.dtype == np.float32

    # 3. Prepare for Wav2Vec2
    prep_w2v2 = SharedAudioPreprocessor.prepare_for_wav2vec2(sine, sample_rate=8000)
    assert len(prep_w2v2) == 16000


def test_shared_preprocessor_rejects_empty_and_silent():
    with pytest.raises(PreprocessingError, match="empty"):
        SharedAudioPreprocessor.decode_and_validate(b"")

    # Silent audio (all zeros)
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    with pytest.raises(PreprocessingError, match="silent"):
        SharedAudioPreprocessor.decode_and_validate(buf.getvalue())


def test_model_a_reports_model_unavailable_when_uncached():
    adapter = ModelA_W2V2AASIST_Adapter(checkpoint_path=None)
    dummy_audio = np.ones(16000, dtype=np.float32) * 0.1
    res = adapter.predict(dummy_audio, 16000)

    assert res["status"] == "unavailable"
    assert res["predicted_class"] == DECISION_UNAVAILABLE
    assert res["spoof_score"] is None
    assert "CHECKPOINT_NOT_CACHED" in res["error"]


def test_model_b_aasist_predicts_valid_contract():
    if not settings.aasist_model_path.is_file():
        pytest.skip("AASIST checkpoint missing.")

    adapter = ModelB_AASIST_Adapter(settings.aasist_model_path)
    # Synthetic burst
    dummy_audio = np.random.uniform(-0.3, 0.3, 32000).astype(np.float32)
    res = adapter.predict(dummy_audio, 16000)

    assert res["status"] == "available"
    assert res["model_id"] == "SpeechAntiSpoofingBenchmarks/AASIST"
    assert res["spoof_score"] is not None
    assert 0.0 <= res["spoof_score"] <= 100.0
    assert res["bona_fide_score"] is not None
    assert round(res["spoof_score"] + res["bona_fide_score"], 1) == 100.0
    assert res["predicted_class"] in ("MODEL_PREDICTS_SPOOF", "MODEL_PREDICTS_BONA_FIDE")


def test_model_c_wav2vec2_predicts_valid_contract():
    adapter = ModelC_Wav2Vec2Deepfake_Adapter()
    dummy_audio = np.random.uniform(-0.3, 0.3, 16000).astype(np.float32)
    res = adapter.predict(dummy_audio, 16000)

    assert res["status"] == "available"
    assert res["model_id"] == "garystafford/wav2vec2-deepfake-voice-detector"
    assert res["spoof_score"] is not None
    assert 0.0 <= res["spoof_score"] <= 100.0
    assert round(res["spoof_score"] + res["bona_fide_score"], 1) == 100.0


def test_audio_heuristics_analyzer_features_and_flags():
    t = np.linspace(0, 1.5, 24000, endpoint=False, dtype=np.float32)
    # Sine wave harmonic
    audio = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)

    res = AudioHeuristicsAnalyzer.analyze(audio, 16000)
    assert res["status"] == "completed"
    assert 0.0 <= res["heuristic_risk_score"] <= 100.0
    assert isinstance(res["quality_flags"], list)
    assert len(res["quality_flags"]) > 0
    assert "rms_energy" in res["supporting_features"]
    assert "spectral_flatness_mean" in res["supporting_features"]


def test_ensemble_threshold_decision_bands():
    ensemble = EnsembleService()

    # Test decision band logic directly on risk scores
    for risk, expected in [
        (45.0, DECISION_LOWER_RISK),
        (69.9, DECISION_LOWER_RISK),
        (70.0, DECISION_REVIEW_REQUIRED),
        (74.5, DECISION_REVIEW_REQUIRED),
        (75.0, DECISION_HIGH_RISK),
        (92.0, DECISION_HIGH_RISK),
    ]:
        if risk < SPOOF_REVIEW_THRESHOLD:
            band = DECISION_LOWER_RISK
        elif risk < SPOOF_HIGH_RISK_THRESHOLD:
            band = DECISION_REVIEW_REQUIRED
        else:
            band = DECISION_HIGH_RISK
        assert band == expected


def test_ensemble_degraded_re_normalization():
    ensemble = EnsembleService()
    dummy_audio = np.random.uniform(-0.2, 0.2, 32000).astype(np.float32)
    res = ensemble.evaluate(dummy_audio, 16000)

    # Since Model A weights are not cached locally, ensemble operates in degraded mode
    assert res["is_degraded"] is True
    assert "SpeechAntiSpoofingBenchmarks/W2V2-AASIST" in res["missing_models"]
    assert "model_b" in res["effective_weights"]
    assert "model_c" in res["effective_weights"]
    assert "heuristics" in res["effective_weights"]
    # Effective weights must sum to 1.0
    assert round(sum(res["effective_weights"].values()), 3) == 1.0


def test_result_service_security_guarantee_phrase_match_never_overrides_spoof():
    quality = {"status": "pass", "speech_usable": True}
    vad = {"status": "completed", "speech_detected": True}
    phrase = {"status": "pass", "exact_match": True, "phrase_verification_status": "pass"}
    features = {"status": "completed"}
    sync = {"status": "unavailable"}

    # 1. High Spoof Risk must result in FAIL even when phrase match is True
    spoof_high_risk = {
        "status": "completed",
        "decision_band": DECISION_HIGH_RISK,
        "predicted_class": DECISION_HIGH_RISK,
        "ensemble_spoof_risk": 88.5,
    }
    res_high = ResultService.build(
        session_id="test-session-1",
        audio_quality=quality,
        speech_detection=vad,
        phrase_verification=phrase,
        feature_extraction=features,
        spoof_detection=spoof_high_risk,
        audio_video_sync=sync,
        started_at=0.0,
    )
    assert res_high["overall_status"] == "fail"
    assert res_high["risk_category"] == "HIGH_RISK"

    # 2. Lower Estimated Risk + Phrase Match True -> PASS
    spoof_low_risk = {
        "status": "completed",
        "decision_band": DECISION_LOWER_RISK,
        "predicted_class": DECISION_LOWER_RISK,
        "ensemble_spoof_risk": 32.0,
    }
    res_low = ResultService.build(
        session_id="test-session-2",
        audio_quality=quality,
        speech_detection=vad,
        phrase_verification=phrase,
        feature_extraction=features,
        spoof_detection=spoof_low_risk,
        audio_video_sync=sync,
        started_at=0.0,
    )
    assert res_low["overall_status"] == "pass"
    assert res_low["risk_category"] == "LOW_RISK"
