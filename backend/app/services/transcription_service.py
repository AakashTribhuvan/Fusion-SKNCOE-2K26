from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import Any


@lru_cache(maxsize=2)
def _load_model(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


class TranscriptionService:
    def __init__(self, model_size: str = "base", language: str = "en") -> None:
        self.model_size = model_size
        self.language = language

    @staticmethod
    def status(model_size: str) -> dict[str, str]:
        if find_spec("faster_whisper") is None:
            return {"status": "unavailable", "details": "Install faster-whisper to enable speech transcription."}
        return {
            "status": "configured",
            "details": f"faster-whisper {model_size} model loads on the first inference request.",
        }

    def transcribe(self, audio, sample_rate: int) -> dict[str, Any]:
        try:
            model = _load_model(self.model_size)
        except (ImportError, OSError, RuntimeError) as exc:
            return {
                "status": "unavailable",
                "recognized_phrase": None,
                "language": None,
                "segments": [],
                "reason": f"faster-whisper model could not be loaded: {exc}",
            }

        try:
            segments, info = model.transcribe(audio, language=self.language, beam_size=1)
            segment_list = list(segments)
            transcript = " ".join(part.text.strip() for part in segment_list if part.text and part.text.strip())
            language = getattr(info, "language", self.language)
            return {
                "status": "completed",
                "recognized_phrase": transcript.strip(),
                "language": language,
                "segments": [{"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()} for seg in segment_list],
                "reason": "Speech recognition completed.",
            }
        except Exception as exc:  # pragma: no cover
            return {
                "status": "error",
                "recognized_phrase": None,
                "language": None,
                "segments": [],
                "reason": f"Transcription failed: {exc}",
            }
