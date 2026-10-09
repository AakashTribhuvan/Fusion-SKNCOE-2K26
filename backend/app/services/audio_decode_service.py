from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import settings


class AudioDecodeService:
    @staticmethod
    def decode_bytes(raw_audio: bytes, sample_rate_hint: int | None = None):
        target_sample_rate = sample_rate_hint or settings.sample_rate
        try:
            audio, sample_rate = sf.read(BytesIO(raw_audio), dtype="float32", always_2d=False)
        except (RuntimeError, OSError):
            try:
                import av

                container = av.open(BytesIO(raw_audio), mode="r")
                audio_stream = next(stream for stream in container.streams if stream.type == "audio")
                resampler = av.AudioResampler(
                    format="flt",
                    layout="mono",
                    rate=target_sample_rate,
                )
                frames = []
                for decoded_frame in container.decode(audio_stream):
                    frames.extend(resampler.resample(decoded_frame))
                frames.extend(resampler.resample(None))
                container.close()
                if not frames:
                    raise ValueError("Audio container contained no decodable samples.")
                audio = np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames])
                sample_rate = target_sample_rate
            except ImportError as exc:
                raise ValueError(
                    "Audio format is not readable by SoundFile; install PyAV to decode browser WebM/Opus recordings."
                ) from exc
            except Exception as exc:
                raise ValueError(
                    "Audio decoding failed. Submit a valid PCM WAV or supported WebM/Opus/MP4 recording."
                ) from exc

        if audio.size == 0:
            raise ValueError("Decoded audio is empty.")

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sample_rate != target_sample_rate:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
                sample_rate = target_sample_rate
            except Exception as exc:
                raise ValueError("Audio resampling failed; please submit a supported PCM WAV recording.") from exc

        return audio.astype(np.float32), int(sample_rate)

    @staticmethod
    def write_temp_wav(audio: np.ndarray, sample_rate: int, target: str | Path) -> str:
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target_path, audio, sample_rate)
        return str(target_path)
