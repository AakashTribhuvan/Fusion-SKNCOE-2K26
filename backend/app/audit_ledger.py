from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from typing import Any


class SimulatedAuditLedger:
    """In-memory hash chain for the demo; this is not a blockchain."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, evidence_hash: str, created_at: datetime) -> dict[str, Any]:
        with self._lock:
            previous_hash = self._entries[-1]["record_hash"] if self._entries else "0" * 64
            record = {
                "sequence": len(self._entries) + 1,
                "previous_hash": previous_hash,
                "evidence_hash": evidence_hash,
                "created_at": created_at.isoformat(),
            }
            record_hash = hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            entry = {**record, "record_hash": record_hash}
            self._entries.append(entry)
            return dict(entry)

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(entry) for entry in self._entries]

    def verify(self) -> bool:
        with self._lock:
            previous_hash = "0" * 64
            for index, entry in enumerate(self._entries, start=1):
                if entry["previous_hash"] != previous_hash:
                    return False
                record = {
                    "sequence": index,
                    "previous_hash": previous_hash,
                    "evidence_hash": entry["evidence_hash"],
                    "created_at": entry["created_at"],
                }
                expected_hash = hashlib.sha256(
                    json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if entry["sequence"] != index or entry["record_hash"] != expected_hash:
                    return False
                previous_hash = expected_hash
            return True
