from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class ChallengeRecord:
    phrase: str
    created_at: datetime
    expires_at: datetime


class ChallengeService:
    _adjectives = [
        "Blue",
        "Green",
        "Silver",
        "Open",
        "Golden",
        "Quiet",
        "Cedar",
        "River",
        "Mountain",
        "Window",
    ]
    _nouns = [
        "river",
        "garden",
        "window",
        "mountain",
        "harbor",
        "forest",
        "station",
        "village",
        "bridge",
        "corner",
    ]
    _numbers = ["seven", "eight", "twenty four", "thirty two", "forty five", "sixty one", "eighty three"]

    @classmethod
    def build_phrase(cls) -> str:
        adjective = secrets.choice(cls._adjectives)
        noun = secrets.choice(cls._nouns)
        number = secrets.choice(cls._numbers)
        return f"{adjective} {noun} {number}".strip()

    @classmethod
    def create(cls, ttl_seconds: int = 900) -> ChallengeRecord:
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(seconds=ttl_seconds)
        return ChallengeRecord(phrase=cls.build_phrase(), created_at=created_at, expires_at=expires_at)
