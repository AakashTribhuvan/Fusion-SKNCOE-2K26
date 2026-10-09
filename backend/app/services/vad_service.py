from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import Any

import numpy as np
import torch


@lru_cache(maxsize=1)
def _load_silero_vad():
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=False)


class VADService:
    @staticmethod
    def status() -> dict[str, str]:
        if find_spec("silero_vad") is None:
            return {"status": "unavailable", "details": "Install silero-vad to enable Silero speech detection."}
        return {"status": "configured", "details": "Silero VAD model loads on the first inference request."}

    def detect(self, audio, sample_rate: int) -> dict[str, Any]:
        try:
            from silero_vad import get_speech_timestamps

            model = _load_silero_vad()
        except (ImportError, OSError, RuntimeError) as exc:
            return {
                "status": "unavailable",
                "speech_detected": False,
                "speech_segments": [],
                "speech_duration_seconds": 0.0,
                "reason": f"Silero VAD could not be loaded: {exc}",
            }

        try:
            samples = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            segments = get_speech_timestamps(
                samples,
                model,
                sampling_rate=sample_rate,
                return_seconds=True,
            )
            speech_segments = [
                {"start": float(segment["start"]), "end": float(segment["end"])}
                for segment in segments
            ]
            total = sum(max(0.0, s["end"] - s["start"]) for s in speech_segments)
            return {
                "status": "completed",
                "speech_detected": bool(speech_segments),
                "speech_segments": speech_segments,
                "speech_duration_seconds": round(total, 4),
                "reason": "Silero VAD completed successfully.",
            }
        except Exception as exc:  # pragma: no cover
            return {
                "status": "error",
                "speech_detected": False,
                "speech_segments": [],
                "speech_duration_seconds": 0.0,
                "reason": f"Silero VAD runtime error: {exc}",
            }
