"""User shell PATH — the bot's systemd env has a minimal PATH (no
~/.npm-global/bin, ~/.local/bin, pyenv, go, texlive, ...), so commands the
agent runs can't see the same toolchain the user's terminal does. Capture the
user's interactive-shell PATH once and inject it into every bash invocation
(and use it for `skills` bin-availability checks).

Captured from the user's SHELL (login+interactive = the exact env the terminal
shows), falling back to bash, then to the inherited PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache

_SHELL_NAMES = ("zsh", "bash")


@lru_cache(maxsize=1)
def _capture_path() -> str:
    inherited = os.environ.get("PATH", "")

    shells: list[str] = []
    configured = (os.environ.get("SHELL") or "").strip()
    if configured and os.path.basename(configured) in _SHELL_NAMES:
        shells.append(configured)
    for name in _SHELL_NAMES:
        which = shutil.which(name)
        if which and which not in shells:
            shells.append(which)

    for shell in shells:
        # Login+interactive first (matches the terminal), then interactive,
        # then plain — each is fast (PATH echo only) and failure falls through.
        for flags in ("-lic", "-ic", "-c"):
            try:
                proc = subprocess.run(
                    [shell, flags, 'echo -n "$PATH"'],
                    capture_output=True, text=True, timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            output = proc.stdout.strip() if proc.returncode == 0 else ""
            if output and output != inherited and "$PATH" not in output:
                return output
    return inherited


def get_user_path() -> str:
    """The user's interactive PATH (cached), or the inherited PATH."""
    return _capture_path()


def with_user_path(command: str) -> str:
    """Prefix a shell command so it sees the user's interactive PATH."""
    user_path = _capture_path()
    inherited = os.environ.get("PATH", "")
    if not user_path or user_path == inherited or "$PATH" in user_path:
        return command
    # Only `:` and `/` appear in PATH entries — safe inside double quotes.
    return f'export PATH="{user_path}:$PATH"; {command}'


def which_in_user_path(name: str) -> str | None:
    """Like shutil.which, but searched with the user's interactive PATH."""
    return shutil.which(name, path=_capture_path())
