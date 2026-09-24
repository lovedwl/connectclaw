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

from connectclaw.security import protected_file_paths


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
            + f"\n\n... （{len(output) - self.max_output_bytes} 字节已截断） ...\n\n"
            + output[-half:]
        )
        return truncated, True


# ── Tier 1: Bubblewrap ────────────────────────────────────────


class BwrapSandbox(Sandbox):
    """
    Full unprivileged container via bubblewrap (bwrap).

    Filesystem model: system paths read-only ($HOME and the real /tmp are
    read-write) — the sandbox contains damage against the OS, not against the
    user's own files. Commands can save anywhere under $HOME or /tmp (curl -o,
    pip --user, npm -g, temp scratch) exactly like a normal shell, and /tmp
    survives across commands (it is the system /tmp, not a per-command tmpfs);
    dangerous commands are gated by BashGuard, not by making the filesystem
    read-only. Network stays open (no --unshare-net) for the same reason.
    """

    @property
    def level(self) -> SandboxLevel:
        return SandboxLevel.BWARP

    async def execute(self, command: str, timeout: int = 120) -> SandboxResult:
        home = os.environ.get("HOME") or os.path.expanduser("~")

        # Build bwrap args: read-only root, read-write $HOME + system /tmp
        bwrap_args = [
            "bwrap",
            "--die-with-parent",
            # Read-only bind entire root — every path is read-only by default;
            # the binds below selectively re-expose writable areas.
            "--ro-bind", "/", "/",
            # Read-write the user's home: curl saves, pip --user, npm -g,
            # project checkouts all work like a normal shell.
            "--bind", home, home,
            # Read-write project directory (defensive; usually under $HOME)
            "--bind", self.cwd, self.cwd,
            # Additional allowed paths (read-write)
            *[arg for path in self.allowed_paths
              for arg in ["--bind", os.path.abspath(path), os.path.abspath(path)]
              if os.path.abspath(path) != self.cwd and os.path.exists(os.path.abspath(path))],
            # Real system /tmp, read-write and persistent across commands
            "--bind", "/tmp", "/tmp",
        ]

        # Escape-hatch security files (config.toml / models.toml) stay
        # READ-ONLY even inside the writable home/cwd — the agent must not
        # modify the operator whitelist or model registry via bash. A missing
        # registry file is touched into existence first, or bash could simply
        # create it (the RO bind only shadows existing files).
        for _pf in protected_file_paths():
            try:
                if not os.path.isfile(_pf):
                    os.makedirs(os.path.dirname(_pf) or ".", exist_ok=True)
                    open(_pf, "a").close()
                bwrap_args += ["--ro-bind", _pf, _pf]
            except OSError:
                continue

        bwrap_args += [
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

    Uses the system /tmp (no private tmpfs) so temp files survive across
    commands. Network stays open — no --net namespace: authorization gates
    danger (BashGuard), not connectivity.
    """

    @property
    def level(self) -> SandboxLevel:
        return SandboxLevel.UNSHARE

    async def execute(self, command: str, timeout: int = 120) -> SandboxResult:
        t0 = time.time()

        unshare_args = ["unshare", "--mount", "--fork", "--pid", "--mount-proc"]

        # Build the inner command
        inner = (
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
                + f"\n\n... （{len(raw_stdout) - max_bytes} 字节已截断） ...\n\n"
                + raw_stdout[-half:]
            )
            result.truncated = True
        else:
            result.stdout = raw_stdout

        result.stderr = raw_stderr[:max_bytes]
        result.exit_code = proc.returncode or 0
        result.wall_time_ms = (time.time() - t0) * 1000

    except FileNotFoundError:
        result.stdout = f"沙箱错误：命令不存在：{cmd_args[0]}"
        result.exit_code = 127
        result.wall_time_ms = (time.time() - t0) * 1000
    except Exception as e:
        result.stdout = f"沙箱错误：{e}"
        result.exit_code = -1
        result.wall_time_ms = (time.time() - t0) * 1000

    return result
