from __future__ import annotations

from typing import Any

import numpy as np


class AudioHeuristicsAnalyzer:
    """Lightweight signal feature extractor and acoustic heuristic risk estimator.

    Extracts classical digital signal processing features (RMS, dynamic range, ZCR,
    spectral centroid, bandwidth, rolloff, flatness, MFCCs, and F0 pitch dynamics).
    Identifies acoustic distortion, unnatural flatness, or recording quality defects.

    IMPORTANT: These heuristic features are auxiliary and provisional. Natural human
    speech and synthetic speech often exhibit overlapping characteristics. Poor audio
    triggers indeterminate review flags rather than synthetic spoof penalties.
    """

    VERSION = "1.0.0-provisional"
    LIMITATIONS = [
        "Acoustic signal features reflect physical acoustic properties and quality indicators, not conclusive proof of AI synthesis or identity.",
        "High-end studio microphones or active noise cancellation can reduce background variance, mimicking synthetic acoustic cleanliness.",
        "Heuristic risk scores are uncalibrated engineering indicators and contribute a small weight to the ensemble.",
        "Corrupted, clipped, or silent recordings trigger quality warnings, not definitive spoof verdicts.",
    ]

    @classmethod
    def analyze(cls, audio: np.ndarray, sample_rate: int = 16000) -> dict[str, Any]:
        waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration_sec = float(len(waveform)) / float(sample_rate) if sample_rate else 0.0

        if waveform.size == 0:
            return {
                "heuristic_risk_score": None,
                "status": "error",
                "quality_flags": ["EMPTY_AUDIO"],
                "supporting_features": {},
                "limitations": cls.LIMITATIONS,
                "heuristic_version": cls.VERSION,
                "reason": "Audio waveform contains no samples.",
            }

        # Energy & dynamic range
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        peak = float(np.max(np.abs(waveform)))
        dyn_range_db = float(20.0 * np.log10((peak + 1e-9) / (rms + 1e-9)))

        # Clipping & silence checks
        clipping_ratio = float(np.mean(np.abs(waveform) >= 0.99))
        silence_flag = rms < 0.001
        clipped_flag = clipping_ratio > 0.01

        quality_flags: list[str] = []
        if silence_flag:
            quality_flags.append("SILENCE_DETECTED")
        if clipped_flag:
            quality_flags.append("CLIPPING_DETECTED")
        if duration_sec < 0.5:
            quality_flags.append("SHORT_DURATION")

        # If audio is silent or invalid, do not inflate spoof score
        if silence_flag:
            return {
                "heuristic_risk_score": 50.0,
                "status": "indeterminate",
                "quality_flags": quality_flags,
                "supporting_features": {
                    "duration_seconds": round(duration_sec, 3),
                    "rms_energy": round(rms, 6),
                    "peak_amplitude": round(peak, 6),
                    "dynamic_range_db": round(dyn_range_db, 2),
                    "clipping_ratio": round(clipping_ratio, 4),
                },
                "limitations": cls.LIMITATIONS,
                "heuristic_version": cls.VERSION,
                "reason": "Silent audio cannot be reliably evaluated for acoustic synthesis.",
            }

        # Classical Spectral Analysis using Librosa
        import librosa

        zcr = librosa.feature.zero_crossing_rate(waveform)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        spec_cent = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
        cent_mean = float(np.mean(spec_cent))
        cent_std = float(np.std(spec_cent))

        spec_bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)
        bw_mean = float(np.mean(spec_bw))

        spec_ro = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)
        ro_mean = float(np.mean(spec_ro))

        spec_flat = librosa.feature.spectral_flatness(y=waveform)
        flat_mean = float(np.mean(spec_flat))
        flat_max = float(np.max(spec_flat))

        mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
        mfcc_means = [round(float(x), 3) for x in np.mean(mfcc, axis=1)]
        mfcc_stds = [round(float(x), 3) for x in np.std(mfcc, axis=1)]

        # Pitch dynamics (YIN)
        pitch_mean = None
        pitch_std = None
        try:
            f0 = librosa.yin(waveform, fmin=60, fmax=400, sr=sample_rate)
            valid_f0 = f0[np.isfinite(f0) & (f0 > 60) & (f0 < 400)]
            if len(valid_f0) > 10:
                pitch_mean = float(np.mean(valid_f0))
                pitch_std = float(np.std(valid_f0))
        except Exception:
            pass

        # Provisional Heuristic Risk Heuristics:
        # Synthetic speech often exhibits:
        # 1. Extremely low dynamic range or abnormally high spectral flatness (buzz/vocoder artifact)
        # 2. Robotic or abnormally low pitch standard deviation in voiced frames (< 10 Hz)
        # 3. Discontinuous high-frequency roll-off
        # Start from a baseline neutral/low risk (e.g. 35.0) and add evidence-based adjustments
        heuristic_score = 40.0

        if flat_mean > 0.05:
            heuristic_score += 15.0
            quality_flags.append("HIGH_SPECTRAL_FLATNESS")
        elif flat_mean < 0.005:
            quality_flags.append("NORMAL_HARMONIC_STRUCTURE")

        if dyn_range_db < 10.0:
            heuristic_score += 10.0
            quality_flags.append("COMPRESSED_DYNAMIC_RANGE")

        if pitch_std is not None and pitch_std < 12.0:
            heuristic_score += 15.0
            quality_flags.append("UNUSUALLY_MONOTONIC_PITCH")
        elif pitch_std is not None and pitch_std > 30.0:
            heuristic_score -= 10.0
            quality_flags.append("NATURAL_PITCH_VARIATION")

        if zcr_mean > 0.25:
            heuristic_score += 10.0
            quality_flags.append("HIGH_FREQUENCY_HISS")

        # Clamp provisional risk to [10.0, 90.0]
        heuristic_score = float(np.clip(heuristic_score, 10.0, 90.0))

        if not quality_flags:
            quality_flags.append("NORMAL_ACOUSTIC_PROFILE")

        return {
            "heuristic_risk_score": round(heuristic_score, 2),
            "status": "completed",
            "quality_flags": quality_flags,
            "supporting_features": {
                "duration_seconds": round(duration_sec, 3),
                "rms_energy": round(rms, 6),
                "peak_amplitude": round(peak, 6),
                "dynamic_range_db": round(dyn_range_db, 2),
                "clipping_ratio": round(clipping_ratio, 4),
                "zero_crossing_rate_mean": round(zcr_mean, 4),
                "zero_crossing_rate_std": round(zcr_std, 4),
                "spectral_centroid_hz": round(cent_mean, 1),
                "spectral_bandwidth_hz": round(bw_mean, 1),
                "spectral_rolloff_hz": round(ro_mean, 1),
                "spectral_flatness_mean": round(flat_mean, 6),
                "pitch_f0_mean_hz": round(pitch_mean, 2) if pitch_mean else None,
                "pitch_f0_std_hz": round(pitch_std, 2) if pitch_std else None,
                "mfcc_mean_1_to_3": mfcc_means[:3],
            },
            "limitations": cls.LIMITATIONS,
            "heuristic_version": cls.VERSION,
            "reason": f"Acoustic feature analysis completed with {len(quality_flags)} flags.",
        }
