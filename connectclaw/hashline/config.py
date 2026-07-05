"""Hashline configuration — lightweight, env-driven, no file I/O."""

from __future__ import annotations

import os
from dataclasses import dataclass

# ─── Constants ─────────────────────────────────────────────────────────────

HASH_LENGTH_MIN = 2
HASH_LENGTH_MAX = 4
DEFAULT_HASH_LENGTH = 2


@dataclass(frozen=True)
class HashlineConfig:
    hash_length: int = DEFAULT_HASH_LENGTH  # 2, 3, or 4
    grep: bool = False


def load_config() -> HashlineConfig:
    """Load config from environment variables, falling back to defaults."""
    hash_length = DEFAULT_HASH_LENGTH
    raw = os.environ.get("HASHLINE_HASH_LENGTH", "")
    if raw:
        try:
            val = int(raw)
            if HASH_LENGTH_MIN <= val <= HASH_LENGTH_MAX:
                hash_length = val
        except ValueError:
            pass

    grep = os.environ.get("HASHLINE_GREP", "0") in ("1", "true", "True")

    return HashlineConfig(hash_length=hash_length, grep=grep)


# Module-level singleton
_config = load_config()


def get_hash_length() -> int:
    return _config.hash_length


def get_grep_enabled() -> bool:
    return _config.grep


def reload_config() -> None:
    """Reload from env (useful for tests)."""
    global _config
    _config = load_config()
