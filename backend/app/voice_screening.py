from __future__ import annotations

import io
import threading
from typing import Any

import av
import numpy as np
import torch


AUDIO_MODEL_ID = "Hemgg/Deepfake-audio-detection"
AUDIO_MODEL_REVISION = "0d75271368ef2c7efd14831dc503c431f6aab0eb"
TRANSCRIPTION_MODEL_ID = "Systran/faster-whisper-tiny.en"
TRANSCRIPTION_MODEL_REVISION = "0d3d19a32d3338f10357c0889762bd8d64bbdeba"
SAMPLE_RATE = 16_000
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
                    if sample_count > SAMPLE_RATE * 10:
                        raise ValueError("Audio must be between 2 and 10 seconds long.")
                    chunks.append(samples)
    except (av.error.FFmpegError, OSError) as error:
        raise ValueError(f"The audio recording could not be decoded: {error}") from error
    if not chunks:
        raise ValueError("The submitted recording contains no decodable audio samples.")
    waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    duration_seconds = len(waveform) / SAMPLE_RATE
    if duration_seconds < 2.0 or duration_seconds > 10.0:
        raise ValueError("Audio must be between 2 and 10 seconds long.")
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
            }
        if _audio_error is not None:
            return {
                "status": "unavailable",
                "model": AUDIO_MODEL_ID,
                "revision": AUDIO_MODEL_REVISION,
                "detail": _audio_error,
            }
        return {
            "status": "not_loaded",
            "model": AUDIO_MODEL_ID,
            "revision": AUDIO_MODEL_REVISION,
            "license": "Apache-2.0",
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


def transcribe_phrase(waveform: np.ndarray) -> str:
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
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
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
        ai_index = next(
            (
                int(index)
                for index, label in labels.items()
                if any(marker in str(label).casefold() for marker in ("ai", "fake", "synthetic", "spoof"))
            ),
            None,
        )
        human_index = next(
            (
                int(index)
                for index, label in labels.items()
                if any(marker in str(label).casefold() for marker in ("human", "real", "bonafide"))
            ),
            None,
        )
        if ai_index is None or human_index is None:
            raise RuntimeError(f"Audio model labels do not identify AI and human classes: {labels!r}")
        return {
            "status": "review",
            "model": AUDIO_MODEL_ID,
            "model_revision": AUDIO_MODEL_REVISION,
            "ai_voice_score": round(float(probabilities[ai_index]), 4),
            "human_voice_score": round(float(probabilities[human_index]), 4),
            "detail": "Uncalibrated research score; it is not proof of a live human speaker.",
        }


def normalize_phrase(text: str) -> str:
    return " ".join("".join(char.lower() if char.isalnum() else " " for char in text).split())


def analyze_audio(audio_bytes: bytes, expected_phrase: str) -> dict[str, Any]:
    waveform = decode_audio(audio_bytes)
    result: dict[str, Any] = {
        "duration_seconds": round(len(waveform) / SAMPLE_RATE, 2),
        "phrase_status": "unavailable",
        "spoof_status": "unavailable",
        "phrase_detail": None,
        "spoof": None,
    }
    try:
        transcription = transcribe_phrase(waveform)
        phrase_matches = normalize_phrase(transcription) == normalize_phrase(expected_phrase)
        result["phrase_status"] = "passed" if phrase_matches else "failed"
        result["phrase_detail"] = (
            "The fresh phrase was transcribed in order."
            if phrase_matches
            else "The transcription did not exactly match the fresh phrase."
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
