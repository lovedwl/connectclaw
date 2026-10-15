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
            out.append(os.path.realpath(os.path.abspath(os.path.expanduser(p))))
    return out


def is_protected(path: str) -> bool:
    # realpath（非 normpath/abspath）：不解析符号链接的话，agent 可以
    # `ln -s ~/.connectclaw ./link` 后经 ./link/config.toml 绕过判定（H2c）。
    ap = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    return any(os.path.realpath(p) == ap for p in protected_file_paths())


def resolve_tool_path(cwd: str, path: str) -> str:
    """Resolve a tool path against cwd; normalize away ``..``/``.``.

    绝对路径也必须 normpath：否则 ``/proj/../proj2/f`` 能以 ``startswith("/proj/")``
    绕过 cwd 前缀检查与 is_protected（M24）。
    """
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(os.path.abspath(cwd), path))


def path_within_cwd(cwd: str, absolute_path: str) -> bool:
    """True if absolute_path resolves inside cwd (symlinks resolved)."""
    cwd_real = os.path.realpath(os.path.abspath(cwd))
    p_real = os.path.realpath(absolute_path)
    return p_real == cwd_real or p_real.startswith(cwd_real + os.sep)
