from __future__ import annotations

from abc import ABC, abstractmethod


class AudioSpoofDetector(ABC):
    """Abstract interface for audio anti-spoofing detection."""

    @abstractmethod
    def predict(self, audio, sample_rate: int):
        raise NotImplementedError
