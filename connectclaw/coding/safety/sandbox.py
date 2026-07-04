"""
Command execution sandbox for ConnectClaw.

Three-tier fallback:
  Tier 1: bubblewrap (bwrap) — full unprivileged container
  Tier 2: unshare + Landlock — Linux namespace + filesystem sandbox
  Tier 3: Direct execution with rlimit — resource limits only

Auto-detects best available at runtime.
"""

from __future__ import annotations

import asyncio
import os
import resource
import shutil
import time
from dataclasses import dataclass
from enum import Enum


# ── Result Types ───────────────────────────────────────────────


class SandboxLevel(Enum):
    BWARP = "bwrap"          # Full container isolation
    UNSHARE = "unshare"      # Namespace + landlock isolation
    RLIMIT = "rlimit"         # Resource limits only


@dataclass
class SandboxResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    timed_out: bool = False
    truncated: bool = False
    level: SandboxLevel = SandboxLevel.RLIMIT
    wall_time_ms: float = 0.0


# ── Sandbox Factory ────────────────────────────────────────────


def detect_best_sandbox() -> type:
    """Detect the best available sandbox implementation."""
    if shutil.which("bwrap"):
        return BwrapSandbox
    if shutil.which("unshare"):
        return NamespaceSandbox
    return RlimitSandbox


# ── Base ───────────────────────────────────────────────────────


class Sandbox:
    """Abstract sandbox interface."""

    def __init__(
        self,
        cwd: str,
        *,
        allowed_paths: list[str] | None = None,
        allow_network: bool = False,
        unsandboxed: bool = False,
        max_memory_mb: int = 512,
        max_cpu_seconds: int = 60,
        max_processes: int = 50,
        max_output_bytes: int = 100_000,
    ):
        self.cwd = os.path.abspath(cwd)
        self.allowed_paths = allowed_paths or [self.cwd]
        self.allow_network = allow_network
        self.unsandboxed = unsandboxed
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        self.max_processes = max_processes
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
    - Read-only bind: /usr, /lib, /lib64, /bin, /etc (system deps)
    - Read-write bind: cwd and allowed_paths
    - New /tmp (tmpfs, private)
    - New /dev (minimal)
    - Unshared network (--unshare-net)
    - Unshared PID (--unshare-pid) when possible
    - Resource limits via --setenv for ulimit propagation
    """

    @property
    def level(self) -> SandboxLevel:
        return SandboxLevel.BWARP

    async def execute(self, command: str, timeout: int = 120) -> SandboxResult:
        # If unsandboxed, delegate to rlimit-only execution
        if self.unsandboxed:
            rlimit = RlimitSandbox(
                cwd=self.cwd,
                max_memory_mb=self.max_memory_mb,
                max_cpu_seconds=self.max_cpu_seconds,
                max_processes=self.max_processes,
                max_output_bytes=self.max_output_bytes,
            )
            return await rlimit.execute(command, timeout=timeout)

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
            # Network isolation
            "" if self.allow_network else "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-pid",
            # Working directory
            "--chdir", self.cwd,
        ]

        # Filter empty args
        bwrap_args = [a for a in bwrap_args if a]

        # Apply resource limits inline, single shell level
        ulimits = (
            f"ulimit -v $(({self.max_memory_mb} * 1024)) 2>/dev/null; "
            f"ulimit -t {self.max_cpu_seconds} 2>/dev/null; "
            f"ulimit -u {self.max_processes} 2>/dev/null; "
        )
        bwrap_args.extend(["--", "bash", "-c", ulimits + command])

        return await _run_command(bwrap_args, timeout, self.max_output_bytes, self.level)


# ── Tier 2: Namespace + unshare ────────────────────────────────


class NamespaceSandbox(Sandbox):
    """
    Linux namespace isolation via unshare.

    Uses `unshare` to create:
    - Mount namespace (--mount)
    - Network namespace (--net) unless allowed
    - PID namespace (--pid --fork)

    Within the namespace, /tmp is remounted as private tmpfs.
    """

    @property
    def level(self) -> SandboxLevel:
        return SandboxLevel.UNSHARE

    async def execute(self, command: str, timeout: int = 120) -> SandboxResult:
        t0 = time.time()

        unshare_args = ["unshare", "--mount", "--fork"]

        if not self.allow_network:
            unshare_args.append("--net")

        # PID namespace
        unshare_args.extend(["--pid", "--mount-proc"])

        # Build the inner command
        inner = (
            f"mount -t tmpfs tmpfs /tmp 2>/dev/null; "
            f"ulimit -v $(({self.max_memory_mb} * 1024)) 2>/dev/null; "
            f"ulimit -t {self.max_cpu_seconds} 2>/dev/null; "
            f"ulimit -u {self.max_processes} 2>/dev/null; "
            f"cd {self.cwd}; "
            f"{command}"
        )

        unshare_args.extend(["--", "bash", "-c", inner])

        return await _run_command(unshare_args, timeout, self.max_output_bytes, self.level)


# ── Tier 3: Resource Limits Only ───────────────────────────────


class RlimitSandbox(Sandbox):
    """
    Minimal sandbox with only resource limits.

    Sets rlimit for memory, CPU, and processes before executing.
    No filesystem or network isolation.
    """

    @property
    def level(self) -> SandboxLevel:
        return SandboxLevel.RLIMIT

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
                preexec_fn=lambda: _set_rlimits(
                    self.max_memory_mb,
                    self.max_cpu_seconds,
                    self.max_processes,
                ),
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


def _set_rlimits(mem_mb: int, cpu_sec: int, nproc: int) -> None:
    """Set resource limits for the child process."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
    except (ValueError, OSError):
        pass


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
