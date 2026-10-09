from __future__ import annotations

from typing import Any

_MAX_CHART_POINTS = 100  # Cap series length sent to the frontend


def _downsample(arr, max_points: int = _MAX_CHART_POINTS):
    """Return at most max_points evenly-spaced values from a 1-D array, safe from NaN."""
    import numpy as np
    arr = np.asarray(arr, dtype=float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) <= max_points:
        return [round(float(x), 4) for x in arr]
    indices = np.linspace(0, len(arr) - 1, max_points, dtype=int)
    return [round(float(x), 4) for x in arr[indices]]


class FeatureExtractionService:
    @staticmethod
    def extract(audio, sample_rate: int, n_mels: int = 40, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
        try:
            import librosa
            import numpy as np
        except Exception as exc:  # pragma: no cover
            return {
                "status": "unavailable",
                "reason": f"librosa is not available: {exc}",
                "feature_names": [],
                "feature_dimensions": {},
            }

        try:
            spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)
            pitch = librosa.yin(audio, fmin=50, fmax=2000, sr=sample_rate)

            # Build time-axis (seconds) at the hop resolution
            n_frames = rms.shape[1]
            frame_times = (np.arange(n_frames) * hop_length / sample_rate).tolist()
            frame_times_ds = _downsample(frame_times)

            # Downsample 1-D series for chart rendering
            rms_ds   = _downsample(rms[0])
            zcr_ds   = _downsample(zcr[0])
            pitch_ds = _downsample(pitch)

            # First 3 MFCC coefficients as separate series
            mfcc_series = {
                f"mfcc_{i+1}": _downsample(mfcc[i]) for i in range(min(3, n_mfcc))
            }

            return {
                "status": "completed",
                "feature_names": ["spectrogram", "mel_spectrogram", "mfcc", "rms_energy", "zero_crossing_rate", "pitch"],
                "feature_dimensions": {
                    "spectrogram": list(spectrogram.shape),
                    "mel_spectrogram": list(mel.shape),
                    "mfcc": list(mfcc.shape),
                    "rms_energy": list(rms.shape),
                    "zero_crossing_rate": list(zcr.shape),
                    "pitch": len(pitch),
                },
                "frame_counts": {
                    "spectrogram": int(spectrogram.shape[1]),
                    "mel_spectrogram": int(mel.shape[1]),
                    "mfcc": int(mfcc.shape[1]),
                    "rms_energy": int(rms.shape[1]),
                    "zero_crossing_rate": int(zcr.shape[1]),
                    "pitch": int(len(pitch)),
                },
                # Time-series arrays (downsampled) for frontend charts
                "chart_time_axis": frame_times_ds,
                "chart_rms":   rms_ds,
                "chart_zcr":   zcr_ds,
                "chart_pitch": _downsample(pitch),
                "chart_mfcc":  mfcc_series,
                "sample_rate": sample_rate,
                "reason": "Feature extraction completed.",
            }
        except Exception as exc:  # pragma: no cover
            return {
                "status": "error",
                "reason": f"Feature extraction failed: {exc}",
                "feature_names": [],
                "feature_dimensions": {},
            }
