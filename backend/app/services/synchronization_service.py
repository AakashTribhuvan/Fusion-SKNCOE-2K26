from __future__ import annotations

import time
from typing import Any

import numpy as np


class SynchronizationService:
    @staticmethod
    def analyze(
        audio_segments: list[dict[str, float]] | None = None,
        video_frames: list[dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        if not audio_segments or not video_frames:
            return {
                "status": "unavailable",
                "timing_offset_seconds": None,
                "analysis_duration_ms": 0,
                "evidence_quality": "missing",
                "reason": "Audio speech segments and Module A mouth-motion measurements are both required.",
            }
        if len(video_frames) < 5:
            return {
                "status": "unavailable",
                "timing_offset_seconds": None,
                "analysis_duration_ms": 0,
                "evidence_quality": "insufficient",
                "reason": "At least five timestamped mouth-motion measurements are required.",
            }

        started = time.perf_counter()
        try:
            timestamps = np.asarray([frame["timestamp_seconds"] for frame in video_frames], dtype=np.float64)
            mouth_motion = np.asarray([frame["mouth_motion"] for frame in video_frames], dtype=np.float64)
            segments = [
                (float(segment["start"]), float(segment["end"]))
                for segment in audio_segments
                if float(segment["end"]) > float(segment["start"])
            ]
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "status": "error",
                "timing_offset_seconds": None,
                "analysis_duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "evidence_quality": "invalid",
                "reason": f"Invalid synchronization evidence: {exc}",
            }

        if not np.isfinite(timestamps).all() or not np.isfinite(mouth_motion).all() or not segments:
            return {
                "status": "error",
                "timing_offset_seconds": None,
                "analysis_duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "evidence_quality": "invalid",
                "reason": "Timestamps, mouth-motion values, and speech segments must be finite and valid.",
            }
        if len(timestamps) != len(mouth_motion) or np.any(np.diff(timestamps) <= 0):
            return {
                "status": "error",
                "timing_offset_seconds": None,
                "analysis_duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "evidence_quality": "invalid",
                "reason": "Video timestamps must be strictly increasing and paired with mouth-motion values.",
            }

        speech_timestamps = np.arange(timestamps[0] - 1.0, timestamps[-1] + 1.0, 0.02)
        speech_activity = np.zeros(speech_timestamps.shape, dtype=np.float64)
        for start, end in segments:
            speech_activity[(speech_timestamps >= start) & (speech_timestamps <= end)] = 1.0
        if np.std(speech_activity) == 0 or np.std(mouth_motion) == 0:
            return {
                "status": "unavailable",
                "timing_offset_seconds": None,
                "analysis_duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "evidence_quality": "insufficient",
                "reason": "Evidence has no measurable variation for offset estimation.",
            }

        offsets = np.arange(-1.0, 1.001, 0.02)
        correlations: list[tuple[float, float]] = []
        for offset in offsets:
            shifted_mouth = np.interp(
                speech_timestamps,
                timestamps + offset,
                mouth_motion,
                left=0.0,
                right=0.0,
            )
            if np.std(shifted_mouth) < 1e-8:
                continue
            correlation = float(np.corrcoef(speech_activity, shifted_mouth)[0, 1])
            if np.isfinite(correlation):
                correlations.append((correlation, float(offset)))

        if not correlations:
            return {
                "status": "unavailable",
                "timing_offset_seconds": None,
                "analysis_duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "evidence_quality": "insufficient",
                "reason": "Could not estimate alignment from the supplied measurements.",
            }

        best_correlation, best_offset = max(correlations)
        return {
            "status": "completed",
            "timing_offset_seconds": round(best_offset, 3),
            "offset_semantics": "Add this offset to video timestamps to align mouth motion with the audio timeline.",
            "alignment_correlation": round(best_correlation, 4),
            "analysis_duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "evidence_quality": "experimental",
            "reason": "Experimental timing estimate only; mismatch is not proof of manipulation.",
        }
