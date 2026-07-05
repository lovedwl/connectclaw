"""
Dynamic tool loader — scan ~/.connectclaw/tools/ for agent-created tools.

Each tool is defined by a .tool.json file:
  {
    "name": "check_types",
    "description": "Check Python type errors",
    "command": "python3 -m pyright {path}",
    "parameters": {
      "path": {"type": "string", "description": "Path to check"}
    }
  }

The agent can create these with the write tool.
They're re-scanned every turn so newly created tools appear immediately.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.coding.safety.sandbox import detect_best_sandbox


class DynamicTool(AgentTool):
    """A tool defined by a .tool.json file. Executes a shell command."""

    def __init__(self, name: str, description: str, command: str, parameters: dict, cwd: str):
        self.name = name
        self.label = name
        self.description = description
        self.parameters = parameters
        self._command = command
        self._cwd = cwd

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        # Format command with params
        try:
            cmd = self._command.format(**params)
        except KeyError as e:
            return AgentToolResult(
                content=[{"type": "text", "text": f"Missing parameter: {e}"}],
            )

        # Execute via sandbox
        sandbox_cls = detect_best_sandbox()
        sandbox = sandbox_cls(cwd=self._cwd, max_memory_mb=256, max_cpu_seconds=120)
        result = await sandbox.execute(cmd, timeout=120)

        output = result.stdout
        if result.timed_out:
            output += "\n\n[timed out]"
        if result.exit_code != 0:
            output += f"\n\n[exit code: {result.exit_code}]"

        return AgentToolResult(
            content=[{"type": "text", "text": output or "(no output)"}],
            details={"command": cmd, "exit_code": result.exit_code},
        )


def load_dynamic_tools(tools_dir: str, cwd: str) -> list[AgentTool]:
    """Scan tools_dir for .tool.json files and return DynamicTool instances.

    Safe to call every turn — only reads .json files, no module loading.
    """
    path = Path(tools_dir)
    if not path.is_dir():
        return []

    tools: list[AgentTool] = []
    for f in sorted(path.glob("*.tool.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        name = data.get("name", f.stem.replace(".tool", ""))
        desc = data.get("description", f"Dynamic tool: {name}")
        command = data.get("command", "")
        parameters = data.get("parameters", {"type": "object", "properties": {}})

        if not command:
            continue

        tools.append(DynamicTool(
            name=name,
            description=desc,
            command=command,
            parameters=parameters,
            cwd=cwd,
        ))

    return tools
