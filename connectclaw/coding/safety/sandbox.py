"""
Command execution sandbox for ConnectClaw.

Design (2026-09): the sandbox provides *isolation*, not resource limits.
Resource limits (RLIMIT_AS in particular) silently break whole toolchains —
node/V8, JVM, Go all reserve large virtual address regions and die on a
512MB cap — while doing nothing to stop a genuinely dangerous command. Risk
control therefore lives in two places:

1. BashGuard: static pattern gate (dangerous → blocked, suspicious → user auth).
2. Sandbox: bwrap / unshare namespaces contain the blast radius (read-only
   root, writable cwd, private /tmp). Network is open by default — per-command
   network authorization was removed; authorization now exists only for
   dangerous commands (BashGuard suspicious) and out-of-cwd writes.

Three-tier fallback:
  Tier 1: bubblewrap (bwrap) — full unprivileged container
  Tier 2: unshare — Linux namespace isolation
  Tier 3: direct execution (no isolation available)

Auto-detects best available at runtime.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass
from enum import Enum


# ── Result Types ───────────────────────────────────────────────


class SandboxLevel(Enum):
    BWARP = "bwrap"          # Full container isolation
    UNSHARE = "unshare"      # Namespace isolation
    DIRECT = "direct"        # No isolation available


@dataclass
class SandboxResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    truncated: bool = False
    level: SandboxLevel = SandboxLevel.DIRECT
    wall_time_ms: float = 0.0


# ── Sandbox Factory ────────────────────────────────────────────


def detect_best_sandbox() -> type:
    """Detect the best available sandbox implementation."""
    if shutil.which("bwrap"):
        return BwrapSandbox
    if shutil.which("unshare"):
        return NamespaceSandbox
    return DirectSandbox


# ── Base ───────────────────────────────────────────────────────


class Sandbox:
    """Abstract sandbox interface."""

    def __init__(
        self,
        cwd: str,
        *,
        allowed_paths: list[str] | None = None,
        max_output_bytes: int = 100_000,
    ):
        self.cwd = os.path.abspath(cwd)
        self.allowed_paths = allowed_paths or [self.cwd]
        self.max_output_bytes = max_output_bytes

    @property
    def level(self) -> SandboxLevel:
        raise NotImplementedError

    async def execute(self, command: str, timeout: int = 120) -> SandboxResult:
        raise NotImplementedError

    def _truncate_output(self, output: str) -> tuple[str, bool]:
        if len(output) <= self.max_output_bytes:
            return output, False
        half = self.max_output_bytes // 2
        truncated = (
            output[:half]
            + f"\n\n... ({len(output) - self.max_output_bytes} bytes truncated) ...\n\n"
            + output[-half:]
        )
        return truncated, True


# ── Tier 1: Bubblewrap ────────────────────────────────────────


class BwrapSandbox(Sandbox):
    """
    Full unprivileged container via bubblewrap (bwrap).

    Creates a new mount namespace with:
    - Read-only bind of the whole root — only cwd and allowed_paths writable
    - New /tmp (tmpfs, private) and minimal /dev
    - Private /proc (PID namespace)
    Network stays open (no --unshare-net): authorization gates a command's
    *danger* (BashGuard), not its connectivity.
    """

    @property
    def level(self) -> SandboxLevel:
        return SandboxLevel.BWARP

    async def execute(self, command: str, timeout: int = 120) -> SandboxResult:
        # Build bwrap args with read-only root + explicit writable paths
        bwrap_args = [
            "bwrap",
            "--die-with-parent",
            # Read-only bind entire root — everything is read-only by default
            "--ro-bind", "/", "/",
            # Read-write project directory
            "--bind", self.cwd, self.cwd,
            # Additional allowed paths (read-write)
            *[arg for path in self.allowed_paths
              for arg in ["--bind", os.path.abspath(path), os.path.abspath(path)]
              if os.path.abspath(path) != self.cwd and os.path.exists(os.path.abspath(path))],
            # Private /tmp on tmpfs (writable, isolated)
            "--tmpfs", "/tmp",
            # Fresh /proc for the PID namespace
            "--proc", "/proc",
            # Minimal /dev
            "--dev", "/dev",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-pid",
            # Working directory
            "--chdir", self.cwd,
        ]

        bwrap_args.extend(["--", "bash", "-c", command])

        return await _run_command(bwrap_args, timeout, self.max_output_bytes, self.level)


# ── Tier 2: Namespace + unshare ────────────────────────────────


class NamespaceSandbox(Sandbox):
    """
    Linux namespace isolation via unshare (mount + PID namespaces).

    /tmp is remounted as a private tmpfs. Network stays open — no --net
    namespace: authorization gates danger (BashGuard), not connectivity.
    """

    @property
    def level(self) -> SandboxLevel:
        return SandboxLevel.UNSHARE

    async def execute(self, command: str, timeout: int = 120) -> SandboxResult:
        t0 = time.time()

        unshare_args = ["unshare", "--mount", "--fork", "--pid", "--mount-proc"]

        # Build the inner command
        inner = (
            f"mount -t tmpfs tmpfs /tmp 2>/dev/null; "
            f"cd {self.cwd}; "
            f"{command}"
        )

        unshare_args.extend(["--", "bash", "-c", inner])

        return await _run_command(unshare_args, timeout, self.max_output_bytes, self.level)


# ── Tier 3: Direct (no isolation available) ────────────────────


class DirectSandbox(Sandbox):
    """
    Fallback when neither bwrap nor unshare is available: plain subprocess.

    Same execution contract (output capture / timeout / truncation), no
    namespace isolation. Safety rests on BashGuard alone.
    """

    @property
    def level(self) -> SandboxLevel:
        return SandboxLevel.DIRECT

    async def execute(self, command: str, timeout: int = 120) -> SandboxResult:
        t0 = time.time()
        result = SandboxResult(level=self.level)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                executable="/bin/bash",
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                result.timed_out = True
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
                result.exit_code = proc.returncode or -1
                result.wall_time_ms = (time.time() - t0) * 1000
                return result

            raw_stdout = stdout_bytes.decode("utf-8", errors="replace")
            raw_stderr = stderr_bytes.decode("utf-8", errors="replace")

            result.stdout, result.truncated = self._truncate_output(raw_stdout)
            result.stderr = raw_stderr[: self.max_output_bytes]
            result.exit_code = proc.returncode or 0
            result.wall_time_ms = (time.time() - t0) * 1000

        except Exception as e:
            result.stdout = str(e)
            result.exit_code = -1
            result.wall_time_ms = (time.time() - t0) * 1000

        return result


# ── Helpers ────────────────────────────────────────────────────


async def _run_command(
    cmd_args: list[str],
    timeout: int,
    max_bytes: int,
    level: SandboxLevel,
) -> SandboxResult:
    """Run a command via asyncio subprocess and collect results."""
    t0 = time.time()
    result = SandboxResult(level=level)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            result.timed_out = True
            try:
                proc.kill()
            except Exception:
                pass
            try:
                # Try to read whatever was captured before timeout
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=3
                )
            except asyncio.TimeoutError:
                await proc.wait()
                stdout_bytes, stderr_bytes = b"", b""
            result.exit_code = proc.returncode or -1
            result.wall_time_ms = (time.time() - t0) * 1000
            # Fall through to decode partial output below

        raw_stdout = stdout_bytes.decode("utf-8", errors="replace")
        raw_stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Truncate
        if len(raw_stdout) > max_bytes:
            half = max_bytes // 2
            result.stdout = (
                raw_stdout[:half]
                + f"\n\n... ({len(raw_stdout) - max_bytes} bytes truncated) ...\n\n"
                + raw_stdout[-half:]
            )
            result.truncated = True
        else:
            result.stdout = raw_stdout

        result.stderr = raw_stderr[:max_bytes]
        result.exit_code = proc.returncode or 0
        result.wall_time_ms = (time.time() - t0) * 1000

    except FileNotFoundError:
        result.stdout = f"sandbox error: command not found: {cmd_args[0]}"
        result.exit_code = 127
        result.wall_time_ms = (time.time() - t0) * 1000
    except Exception as e:
        result.stdout = f"sandbox error: {e}"
        result.exit_code = -1
        result.wall_time_ms = (time.time() - t0) * 1000

    return result
