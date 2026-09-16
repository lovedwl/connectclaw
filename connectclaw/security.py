"""Paths the agent must NEVER modify — the escape-hatch security config.

The ``!``/``/bash`` operator whitelist lives in config.toml (``[bash]
operator_open_ids``) and the model registry in models.toml; those two files are
the lock of the escape hatches. The agent can write files (write / hash_edit)
and run bash, so both layers are closed here:
  - write / hash_edit refuse the paths outright;
  - the bwrap sandbox mounts them read-only even inside the writable cwd/home,
    so bash cannot touch them either.

Anything that would let the agent widen its own whitelist voids the escapes.
"""

from __future__ import annotations

import os

from connectclaw.config import _find_config_path
from connectclaw.model_registry import DEFAULT_REGISTRY


def protected_file_paths() -> list[str]:
    """Absolute paths the agent may not read-write (config + registry)."""
    out: list[str] = []
    for p in (_find_config_path(), DEFAULT_REGISTRY):
        if p:
            out.append(os.path.abspath(os.path.expanduser(p)))
    return out


def is_protected(path: str) -> bool:
    ap = os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    return any(os.path.normpath(p) == ap for p in protected_file_paths())
