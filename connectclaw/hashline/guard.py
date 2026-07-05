"""Noop loop guard — prevents models from looping on identical edits.

Ported from pi-hashline-edit/src/noop-loop-guard.ts (MIT).

Three consecutive byte-identical no-op edit payloads on the same content
throw [E_NOOP_LOOP], breaking the cycle.
"""

from __future__ import annotations

NOOP_HARD_LIMIT = 3

_noop_tracker: dict[str, dict] = {}  # path → {payloadKey, count}
_applied_payload_tracker: dict[str, str] = {}  # path → payloadKey


def record_noop_edit(path: str, payload_key: str) -> tuple[int, bool]:
    """Record a noop edit attempt. Returns (count, escalate)."""
    existing = _noop_tracker.get(path)
    if existing and existing["payloadKey"] == payload_key:
        existing["count"] += 1
    else:
        _noop_tracker[path] = {"payloadKey": payload_key, "count": 1}

    count = _noop_tracker[path]["count"]
    return (count, count >= NOOP_HARD_LIMIT)


def record_applied_edit(path: str, payload_key: str) -> None:
    """Record a successfully applied edit payload key."""
    _noop_tracker.pop(path, None)
    _applied_payload_tracker[path] = payload_key


def is_duplicate_applied_payload(path: str, payload_key: str) -> bool:
    """Check if this payload was already successfully applied."""
    return _applied_payload_tracker.get(path) == payload_key


def clear_applied_payload(path: str) -> None:
    """Clear the applied-payload record — called on deliberate re-read."""
    _applied_payload_tracker.pop(path, None)


def reset_noop_guard() -> None:
    """Reset all counters — for tests only."""
    _noop_tracker.clear()
    _applied_payload_tracker.clear()
