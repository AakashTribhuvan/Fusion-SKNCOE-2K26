from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

AASIST_SAMPLE_RATE = 16000
AASIST_FIXED_SAMPLES = 64600  # ~4.0375s at 16kHz per official ASVspoof protocol
WAV2VEC2_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.3
MAX_DURATION_SECONDS = 10.0
SILENCE_RMS_THRESHOLD = 0.001  # -60 dBFS


class PreprocessingError(Exception):
    """Raised when audio fails quality, silence, or decoding constraints."""
    pass


class SharedAudioPreprocessor:
    """Centralized, deterministic audio preprocessing service for multi-model anti-spoofing."""

    @staticmethod
    def decode_and_validate(
        raw_bytes: bytes,
        min_seconds: float = MIN_DURATION_SECONDS,
        max_seconds: float = MAX_DURATION_SECONDS,
    ) -> tuple[np.ndarray, int]:
        """Decode raw audio bytes, convert to mono float32, and validate duration and energy."""
        if not raw_bytes or len(raw_bytes) == 0:
            raise PreprocessingError("Audio payload is empty (0 bytes).")

        try:
            audio, sample_rate = sf.read(BytesIO(raw_bytes), dtype="float32", always_2d=False)
        except Exception:
            try:
                import av

                container = av.open(BytesIO(raw_bytes), mode="r")
                audio_stream = next(s for s in container.streams if s.type == "audio")
                resampler = av.AudioResampler(format="flt", layout="mono", rate=AASIST_SAMPLE_RATE)
                frames = []
                for frame in container.decode(audio_stream):
                    frames.extend(resampler.resample(frame))
                frames.extend(resampler.resample(None))
                container.close()
                if not frames:
                    raise PreprocessingError("Audio container contains no decodable audio frames.")
                audio = np.concatenate([f.to_ndarray().reshape(-1) for f in frames])
                sample_rate = AASIST_SAMPLE_RATE
            except Exception as exc:
                raise PreprocessingError(f"Audio decoding failed: {exc}") from exc

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise PreprocessingError("Decoded audio contains no samples.")

        # Convert stereo or multi-channel to mono deterministically
        if audio.ndim == 2:
            audio = np.mean(audio, axis=-1)
        audio = audio.reshape(-1)

        duration = float(len(audio)) / float(sample_rate) if sample_rate else 0.0
        if duration < min_seconds:
            raise PreprocessingError(
                f"Audio duration ({duration:.2f}s) is shorter than minimum required ({min_seconds:.2f}s)."
            )
        if duration > max_seconds:
            raise PreprocessingError(
                f"Audio duration ({duration:.2f}s) exceeds maximum allowed ({max_seconds:.2f}s)."
            )

        # Silence check
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < SILENCE_RMS_THRESHOLD:
            raise PreprocessingError(
                f"Audio is silent or nearly silent (RMS={rms:.6f} < threshold={SILENCE_RMS_THRESHOLD:.6f})."
            )

        return audio, int(sample_rate)

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Deterministically resample waveform using torchaudio sinc interpolation."""
        if orig_sr == target_sr:
            return np.asarray(audio, dtype=np.float32)

        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        resampled = F.resample(tensor, orig_freq=orig_sr, new_freq=target_sr)
        return resampled.numpy().astype(np.float32)

    @classmethod
    def prepare_for_aasist(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        target_samples: int = AASIST_FIXED_SAMPLES,
    ) -> np.ndarray:
        """Prepare audio for AASIST / W2V2-AASIST: 16 kHz mono, exactly 64,600 samples with tiling."""
        resampled = cls.resample(audio, sample_rate, AASIST_SAMPLE_RATE)
        if resampled.size >= target_samples:
            return resampled[:target_samples]

        # Official ASVspoof evaluation procedure: deterministic tile repetition for short clips
        repeats = (target_samples // resampled.size) + 1
        return np.tile(resampled, repeats)[:target_samples]

    @classmethod
    def prepare_for_wav2vec2(
        cls,
        audio: np.ndarray,
        sample_rate: int,
        max_samples: int = 160000,  # 10s at 16kHz
    ) -> np.ndarray:
        """Prepare audio for Wav2Vec2 audio classification: 16 kHz mono waveform."""
        resampled = cls.resample(audio, sample_rate, WAV2VEC2_SAMPLE_RATE)
        if resampled.size > max_samples:
            return resampled[:max_samples]
        return resampled
