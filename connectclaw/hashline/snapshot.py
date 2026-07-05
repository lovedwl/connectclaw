"""Per-path multi-version LRU snapshot store for stale-anchor recovery.

Ported from pi-hashline-edit/src/read-snapshot.ts (MIT).

Memory bounds:
  MAX_PATHS (8) × MAX_VERSIONS_PER_PATH (4) entries,
  plus a total UTF-16 length cap (~32 MiB).
"""

from __future__ import annotations

MAX_PATHS = 8
MAX_VERSIONS_PER_PATH = 4
# 32 MiB soft ceiling measured in Python string length (≈ UTF-16 code units for BMP)
MAX_TOTAL_CHARS = 32 * 1024 * 1024

# Paths stored in MRU-first order (index 0 = most recently used).
_path_order: list[str] = []
_path_map: dict[str, list[str]] = {}  # path → versions (newest first)


def _total_size() -> int:
    return sum(len(v) for entry in _path_map.values() for v in entry)


def _evict_oldest_version() -> None:
    """Evict the oldest version of the least-recently-used path."""
    for i in range(len(_path_order) - 1, -1, -1):
        p = _path_order[i]
        entry = _path_map.get(p)
        if entry and entry:
            entry.pop()  # remove oldest (last in newest-first list)
            if not entry:
                del _path_map[p]
                _path_order.pop(i)
            return


def remember_read_snapshot(canonical_path: str, content: str) -> None:
    """Record a hashline read snapshot for canonicalPath.

    - Byte-identical to current newest → no-op (read fusion).
    - Moves path to MRU position on every non-fused write.
    - Evicts oldest versions/paths to stay within limits.
    """
    existing = _path_map.get(canonical_path)

    # Read fusion: skip if identical to most recent version.
    if existing and existing and existing[0] == content:
        # Promote to MRU
        if canonical_path in _path_order:
            idx = _path_order.index(canonical_path)
            if idx > 0:
                _path_order.pop(idx)
                _path_order.insert(0, canonical_path)
        return

    if existing:
        existing.insert(0, content)  # prepend newest
        while len(existing) > MAX_VERSIONS_PER_PATH:
            existing.pop()
        # Move to MRU
        if canonical_path in _path_order:
            idx = _path_order.index(canonical_path)
            if idx > 0:
                _path_order.pop(idx)
                _path_order.insert(0, canonical_path)
    else:
        # New path: evict LRU if at limit
        if len(_path_order) >= MAX_PATHS:
            lru = _path_order[-1]
            del _path_map[lru]
            _path_order.pop()
        _path_map[canonical_path] = [content]
        _path_order.insert(0, canonical_path)

    # Byte-budget eviction
    while _total_size() > MAX_TOTAL_CHARS:
        _evict_oldest_version()
        if not _path_map:
            break


def get_read_snapshot(canonical_path: str) -> str | None:
    """Return the most recent snapshot for canonicalPath, or None."""
    entry = _path_map.get(canonical_path)
    return entry[0] if entry else None


def get_read_snapshot_versions(canonical_path: str) -> list[str]:
    """Return all stored versions in newest-first order."""
    entry = _path_map.get(canonical_path)
    return list(entry) if entry else []


def reset_read_snapshots() -> None:
    """Reset the entire store — for tests only."""
    _path_order.clear()
    _path_map.clear()
