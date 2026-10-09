from __future__ import annotations

import difflib
import io
import threading
from collections import Counter
from typing import Any

import av
import numpy as np
import torch


AUDIO_MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
AUDIO_MODEL_REVISION = "c66306024a7ede0be291e9c4558b37634782dc4e"
TRANSCRIPTION_MODEL_ID = "Systran/faster-whisper-tiny.en"
TRANSCRIPTION_MODEL_REVISION = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
SAMPLE_RATE = 16_000
AUDIO_MIN_DURATION_SECONDS = 0.3
AUDIO_MAX_DURATION_SECONDS = 10.0
AUDIO_SILENCE_RMS_THRESHOLD = 0.001
PHRASE_MATCH_THRESHOLD = 0.90
_model_lock = threading.RLock()
_audio_processor: Any = None
_audio_model: Any = None
_audio_error: str | None = None
_whisper: Any = None
_whisper_error: str | None = None


def decode_audio(audio_bytes: bytes) -> np.ndarray:
    try:
        with av.open(io.BytesIO(audio_bytes)) as container:
            if not container.streams.audio:
                raise ValueError("The submitted recording contains no audio stream.")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            chunks: list[np.ndarray] = []
            sample_count = 0
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    samples = resampled.to_ndarray().reshape(-1)
                    sample_count += samples.size
                    if sample_count > SAMPLE_RATE * AUDIO_MAX_DURATION_SECONDS:
                        raise ValueError("Audio must be between 0.3 and 10 seconds long.")
                    chunks.append(samples)
    except (av.error.FFmpegError, OSError) as error:
        raise ValueError(f"The audio recording could not be decoded: {error}") from error
    if not chunks:
        raise ValueError("The submitted recording contains no decodable audio samples.")
    waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    duration_seconds = len(waveform) / SAMPLE_RATE
    if duration_seconds < AUDIO_MIN_DURATION_SECONDS or duration_seconds > AUDIO_MAX_DURATION_SECONDS:
        raise ValueError("Audio must be between 0.3 and 10 seconds long.")
    if not np.isfinite(waveform).all():
        raise ValueError("The submitted recording contains invalid audio samples.")
    return waveform


def _load_audio_model() -> None:
    global _audio_processor, _audio_model, _audio_error
    if _audio_model is not None:
        return
    if _audio_error is not None:
        raise RuntimeError(_audio_error)
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        processor = AutoFeatureExtractor.from_pretrained(
            AUDIO_MODEL_ID,
            revision=AUDIO_MODEL_REVISION,
        )
        model = AutoModelForAudioClassification.from_pretrained(
            AUDIO_MODEL_ID,
            revision=AUDIO_MODEL_REVISION,
        )
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
    except Exception as error:
        _audio_error = f"Audio anti-spoof model could not be loaded: {type(error).__name__}: {error}"
        raise RuntimeError(_audio_error) from error
    _audio_processor = processor
    _audio_model = model


def audio_model_status() -> dict[str, str]:
    with _model_lock:
        if _audio_model is not None:
            return {
                "status": "ready",
                "model": AUDIO_MODEL_ID,
                "revision": AUDIO_MODEL_REVISION,
                "license": "Apache-2.0",
                "weights_size": "approximately 1.2 GB",
            }
        if _audio_error is not None:
            return {
                "status": "unavailable",
                "model": AUDIO_MODEL_ID,
                "revision": AUDIO_MODEL_REVISION,
                "license": "Apache-2.0",
                "detail": _audio_error,
            }
        return {
            "status": "not_loaded",
            "model": AUDIO_MODEL_ID,
            "revision": AUDIO_MODEL_REVISION,
            "license": "Apache-2.0",
            "weights_size": "approximately 1.2 GB",
        }


def warm_audio_model() -> dict[str, str]:
    with _model_lock:
        try:
            _load_audio_model()
        except RuntimeError:
            return audio_model_status()
        return audio_model_status()


def warm_transcriber() -> dict[str, str]:
    global _whisper, _whisper_error
    try:
        with _model_lock:
            if _whisper is None:
                from faster_whisper import WhisperModel

                _whisper = WhisperModel(
                    TRANSCRIPTION_MODEL_ID,
                    device="cpu",
                    compute_type="int8",
                    download_root=".tools/models/whisper",
                    revision=TRANSCRIPTION_MODEL_REVISION,
                )
        return {
            "status": "ready",
            "model": TRANSCRIPTION_MODEL_ID,
            "revision": TRANSCRIPTION_MODEL_REVISION,
            "license": "MIT",
        }
    except Exception as error:
        _whisper_error = f"Phrase transcription is unavailable: {type(error).__name__}: {error}"
        return {
            "status": "unavailable",
            "model": TRANSCRIPTION_MODEL_ID,
            "revision": TRANSCRIPTION_MODEL_REVISION,
            "detail": _whisper_error,
        }


def transcribe_phrase(waveform: np.ndarray, expected_phrase: str) -> str:
    global _whisper, _whisper_error
    if _whisper_error is not None:
        raise RuntimeError(_whisper_error)
    try:
        with _model_lock:
            if _whisper is None:
                status = warm_transcriber()
                if status["status"] != "ready":
                    raise RuntimeError(status["detail"])
            segments, _ = _whisper.transcribe(
                waveform,
                language="en",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=False,
                hotwords=expected_phrase,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        _whisper_error = f"Phrase transcription is unavailable: {type(error).__name__}: {error}"
        raise RuntimeError(_whisper_error) from error


def classify_audio(waveform: np.ndarray) -> dict[str, Any]:
    with _model_lock:
        _load_audio_model()
        inputs = _audio_processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        device = next(_audio_model.parameters()).device
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(_audio_model(**inputs).logits, dim=-1)[0]
        labels = _audio_model.config.id2label
        label_indices = {str(label).strip().casefold(): int(index) for index, label in labels.items()}
        ai_index = label_indices.get("fake")
        human_index = label_indices.get("real")
        if ai_index is None or human_index is None:
            raise RuntimeError(f"Audio model labels do not identify real and fake classes: {labels!r}")
        ai_score = round(float(probabilities[ai_index]), 4)
        human_score = round(float(probabilities[human_index]), 4)
        return {
            "status": "review",
            "model": AUDIO_MODEL_ID,
            "model_revision": AUDIO_MODEL_REVISION,
            "license": "Apache-2.0",
            "ai_voice_score": ai_score,
            "human_voice_score": human_score,
            "detail": (
                f"Uncalibrated research score: {ai_score:.1%} synthetic-like and "
                f"{human_score:.1%} human-like. This is not proof of a live human speaker."
            ),
        }


def normalize_phrase(text: str) -> str:
    return " ".join("".join(char.lower() if char.isalnum() else " " for char in text).split())


def compare_phrase(expected_phrase: str, transcription: str) -> dict[str, Any]:
    expected = normalize_phrase(expected_phrase)
    recognized = normalize_phrase(transcription)
    if not expected or not recognized:
        return {"status": "failed", "similarity": 0.0, "exact_match": False}

    expected_tokens = expected.split()
    recognized_tokens = recognized.split()
    overlap = sum((Counter(expected_tokens) & Counter(recognized_tokens)).values())
    token_similarity = overlap / max(len(expected_tokens), len(recognized_tokens))
    text_similarity = difflib.SequenceMatcher(None, expected, recognized).ratio()
    similarity = round(max(token_similarity, text_similarity), 4)
    return {
        "status": "passed" if similarity >= PHRASE_MATCH_THRESHOLD else "failed",
        "similarity": similarity,
        "exact_match": expected == recognized,
    }


def _audio_quality(waveform: np.ndarray) -> dict[str, Any]:
    duration_seconds = len(waveform) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    peak = float(np.max(np.abs(waveform)))
    clipping_ratio = float(np.mean(np.abs(waveform) >= 0.99))
    silent = rms < AUDIO_SILENCE_RMS_THRESHOLD
    clipped = clipping_ratio > 0.01
    short_for_model = duration_seconds < 2.5

    if silent:
        status = "failed"
        detail = (
            f"Recording is silent or nearly silent (RMS {rms:.6f}; "
            f"minimum {AUDIO_SILENCE_RMS_THRESHOLD:.3f}). Check the microphone and retry."
        )
    elif clipped or short_for_model:
        status = "review"
        reasons = []
        if clipped:
            reasons.append("the signal is clipped or distorted")
        if short_for_model:
            reasons.append("the clip is shorter than the model's stated 2.5-second optimal range")
        detail = f"Audio quality needs review because {' and '.join(reasons)}."
    else:
        status = "passed"
        detail = "Audio duration and signal level are suitable for this research screening."

    return {
        "status": status,
        "detail": detail,
        "duration_seconds": round(duration_seconds, 2),
        "sample_rate": SAMPLE_RATE,
        "peak_amplitude": round(peak, 6),
        "rms_energy": round(rms, 6),
        "clipping_ratio": round(clipping_ratio, 4),
        "silent": silent,
        "clipped": clipped,
    }


def analyze_audio(audio_bytes: bytes, expected_phrase: str) -> dict[str, Any]:
    waveform = decode_audio(audio_bytes)
    quality = _audio_quality(waveform)
    result: dict[str, Any] = {
        "duration_seconds": round(len(waveform) / SAMPLE_RATE, 2),
        "phrase_status": "unavailable",
        "spoof_status": "unavailable",
        "phrase_detail": None,
        "phrase_similarity": None,
        "quality": quality,
        "spoof": None,
    }
    if quality["status"] == "failed":
        result["phrase_detail"] = quality["detail"]
        result["spoof"] = {"status": "unavailable", "detail": quality["detail"]}
        return result

    try:
        transcription = transcribe_phrase(waveform, expected_phrase)
        phrase_result = compare_phrase(expected_phrase, transcription)
        result["phrase_status"] = phrase_result["status"]
        result["phrase_similarity"] = phrase_result["similarity"]
        result["phrase_detail"] = (
            (
                "The fresh phrase was transcribed exactly."
                if phrase_result["exact_match"]
                else f"The fresh phrase was transcribed with {phrase_result['similarity']:.0%} similarity."
            )
            if phrase_result["status"] == "passed"
            else f"The transcription matched the fresh phrase with {phrase_result['similarity']:.0%} similarity; 90% is required."
        )
    except RuntimeError as error:
        result["phrase_detail"] = str(error)

    try:
        spoof = classify_audio(waveform)
        result["spoof_status"] = spoof["status"]
        result["spoof"] = spoof
    except RuntimeError as error:
        result["spoof"] = {"status": "unavailable", "detail": str(error)}
    return result
