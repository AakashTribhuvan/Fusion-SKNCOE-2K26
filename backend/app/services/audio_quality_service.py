from __future__ import annotations

import numpy as np


class AudioQualityService:
    @staticmethod
    def analyze(
        audio: np.ndarray,
        sample_rate: int,
        min_duration_seconds: float = 0.3,
        silence_threshold: float = 0.001,   # RMS in float32 [-1, 1] space.
        # 0.001 ≈ -60 dBFS — genuine speech, even from a low-gain USB mic
        # recorded via WebM→WAV transcoding, reliably exceeds this.
        # The old default (0.01) was incorrectly rejecting real speech.
    ):
        duration_seconds = float(len(audio)) / float(sample_rate) if sample_rate else 0.0
        if len(audio) == 0:
            return {
                "status": "fail",
                "reason": "Decoded audio contains no samples.",
                "duration_seconds": 0.0,
                "sample_rate": sample_rate,
                "speech_usable": False,
            }

        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        silent = bool(rms < silence_threshold)
        clipped = bool(peak > 0.99)

        status = "pass"
        reason = "Audio quality checks passed."
        if duration_seconds < min_duration_seconds:
            status = "fail"
            reason = "Recording is excessively short."
        elif silent:
            status = "fail"
            reason = (
                f"Recording is silent or nearly silent (RMS={rms:.6f}, threshold={silence_threshold:.6f}). "
                "Ensure the microphone is not muted and the correct input device is selected."
            )
        elif clipped:
            # Clipped audio still contains speech — mark as suspicious but
            # allow the pipeline to continue. Do NOT short-circuit VAD.
            status = "suspicious"
            reason = "Recording appears clipped or distorted (peak ≥ 0.99). Analysis will continue."

        return {
            "status": status,
            "duration_seconds": round(duration_seconds, 4),
            "sample_rate": sample_rate,
            "channels": 1,
            "samples": int(len(audio)),
            "peak_amplitude": round(peak, 6),
            "rms_energy": round(rms, 6),
            "silence_threshold_used": silence_threshold,
            "silent": silent,
            "clipped": clipped,
            "speech_usable": status in ("pass", "suspicious"),
            "reason": reason,
        }
