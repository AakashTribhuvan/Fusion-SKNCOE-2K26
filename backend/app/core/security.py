from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone


def generate_session_id() -> str:
    return secrets.token_urlsafe(16)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_ts(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def is_session_expired(created_at: datetime, ttl_seconds: int) -> bool:
    return utc_now() > created_at + timedelta(seconds=ttl_seconds)
