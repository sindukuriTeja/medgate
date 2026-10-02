"""Hash-chained, tamper-evident audit trail.

Every gated decision, license event, and post-market observation is appended
to a chain where each entry's hash covers the previous entry's hash plus the
entry's canonical content. Any later tampering breaks the chain, which is what
makes an authorization *contestable*: a reviewer can re-verify the whole
history offline and prove whether a decision was altered after the fact.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class AuditTrail:
    """Append-only, hash-chained log of gate and license events."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._prev_hash = "0" * 64  # genesis

    def append(self, event: str, data: Dict[str, Any], at: str = "t?") -> Dict[str, Any]:
        seq = len(self._entries)
        body = {
            "seq": seq,
            "at": at,
            "event": event,
            "data": data,
            "prev_hash": self._prev_hash,
        }
        digest = hashlib.sha256(
            (self._prev_hash + _canonical(body)).encode("utf-8")
        ).hexdigest()
        entry = dict(body, hash=digest)
        self._prev_hash = digest
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        """Recompute the whole chain; True iff no entry was altered."""
        prev = "0" * 64
        for entry in self._entries:
            body = {
                "seq": entry["seq"],
                "at": entry["at"],
                "event": entry["event"],
                "data": entry["data"],
                "prev_hash": entry["prev_hash"],
            }
            if body["prev_hash"] != prev:
                return False
            digest = hashlib.sha256(
                (prev + _canonical(body)).encode("utf-8")
            ).hexdigest()
            if digest != entry["hash"]:
                return False
            prev = digest
        return True

    def entries(self) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def head_hash(self) -> Optional[str]:
        return self._entries[-1]["hash"] if self._entries else None