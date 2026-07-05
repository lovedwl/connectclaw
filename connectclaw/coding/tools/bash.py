"""Bash tool — execute shell commands with safety checks."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Literal

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.coding.safety.sandbox import detect_best_sandbox


class BashGuard:
    """Pattern-based dangerous command detection.

    Two tiers:
    - DANGEROUS: always blocked (rm -rf /, mkfs, dd raw, fork bomb, etc.)
    - SUSPICIOUS: require Feishu card authorization (rm, mv, chmod, eval, curl pipe)
    """

    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/(?:\s|$)",    # rm -rf / (root only, followed by space or end)
        r"rm\s+-rf\s+~\s*$",        # rm -rf ~ (home only, followed by space or end)
        r"mkfs\.",                   # format filesystem
        r"dd\s+if=.*of=/dev/",      # dd to device
        r">\s*/dev/sd[a-z]",        # redirect to raw device
        r":\(\)\s*\{",              # fork bomb
        r"shutdown\b",              # shutdown
        r"reboot\b",                # reboot
        r"halt\b",                  # halt
        r"poweroff\b",              # poweroff
        r"iptables\b",              # firewall modification
        r"systemctl\s+stop\b",      # stop system services
        r"systemctl\s+disable\b",   # disable system services
    ]

    SUSPICIOUS_PATTERNS = [
        r"\brm\b",                 # any rm
        r"\bmv\b.*/(?:etc|usr|var|opt|bin|sbin|lib|boot)",  # mv to system dir
        r"\bchmod\s+[0-7]*7[0-7]*[0-7]*\b",  # chmod with write/exec
        r"\bchown\b",              # chown
        r"\beval\b",               # eval
        r"\bexec\b",               # exec
        r"\bsource\b",             # source
        r"curl.*\|.*(?:ba)?sh",   # curl pipe to shell
        r"wget.*\|.*(?:ba)?sh",   # wget pipe to shell
        r"\bgit\s+push\s+--force", # force push
        r"\bgit\s+push\s+-f\b",   # force push short
        r"\bdocker\s+rm\b",       # docker remove
        r"\bdocker\s+system\s+prune",  # docker prune
        r"\bnpm\s+publish\b",    # npm publish
        r"\bpip\s+uninstall\b",  # pip uninstall
    ]

    def check(self, command: str) -> Literal["SAFE", "SUSPICIOUS", "DANGEROUS"]:
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return "DANGEROUS"
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return "SUSPICIOUS"
        return "SAFE"


class BashTool(AgentTool):
    name = "bash"
    label = "bash"
    description = (
        "Execute a shell command in a subprocess. "
        "The command will be executed in the working directory. "
        "A timeout can be specified (default 120 seconds). "
        "Some commands may require user authorization."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 120, max: 600)",
            },
            "allow_network": {
                "type": "boolean",
                "description": "Allow network access for this command (requires user approval). Use when git push/pull, curl, pip install, npm install, etc.",
            },
            "unsandboxed": {
                "type": "boolean",
                "description": "Run outside the sandbox entirely (requires user approval). Use only when the sandbox blocks essential functionality.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, cwd: str, guard: BashGuard | None = None):
        self._cwd = cwd
        self._guard = guard or BashGuard()
        self._sandbox_cls = detect_best_sandbox()

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        command = params["command"]
        timeout = min(params.get("timeout", 120) or 120, 600)

        # Safety check
        check = self._guard.check(command)
        if check == "DANGEROUS":
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"Command blocked for safety: `{command}`\n"
                        f"This command matches dangerous patterns and cannot be executed."
                    ),
                }],
            )

        # Execute in sandbox (with optional network or full escape)
        allow_network = params.get("allow_network", False)
        unsandboxed = params.get("unsandboxed", False)

        sandbox = self._sandbox_cls(
            cwd=self._cwd,
            allow_network=allow_network,
            unsandboxed=unsandboxed,
            max_memory_mb=512,
            max_cpu_seconds=timeout,
            max_processes=50,
        )

        result = await sandbox.execute(command, timeout=timeout)

        # Format output
        output = result.stdout
        if result.timed_out:
            output = f"Command timed out after {timeout}s: `{command}`\n\nPartial output:\n{output}"

        details = {
            "exit_code": result.exit_code,
            "command": command,
            "sandbox_level": result.level.value,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
        }

        return AgentToolResult(
            content=[{"type": "text", "text": output or "(no output)"}],
            details=details,
        )


def create_bash_tool(cwd: str, guard: BashGuard | None = None) -> BashTool:
    return BashTool(cwd, guard)
