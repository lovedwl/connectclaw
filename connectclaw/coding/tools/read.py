"""Read tool — reads file contents with line numbers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles

from connectclaw.agent.types import AgentTool, AgentToolResult


class ReadTool(AgentTool):
    name = "read"
    label = "read"
    description = (
        "Read the contents of a file from the filesystem. "
        "Returns the file content with line numbers. "
        "Supports offset and limit for reading specific ranges."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, cwd: str):
        self._cwd = cwd
        # Track recently read files for write-before-read enforcement
        self.recently_read: set[str] = set()

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        file_path = params["file_path"]
        absolute_path = self._resolve_path(file_path)

        # Check file exists
        if not os.path.isfile(absolute_path):
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error: File not found: {file_path}"}],
            )

        # Read file
        try:
            async with aiofiles.open(absolute_path, "r") as f:
                content = await f.read()
        except Exception as e:
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error reading file: {e}"}],
            )

        lines = content.split("\n")
        total_lines = len(lines)

        # Apply offset and limit
        start = (params.get("offset", 1) or 1) - 1
        start = max(0, start)
        limit = params.get("limit")

        if limit is not None:
            selected = lines[start : start + limit]
        else:
            selected = lines[start:]

        # Mark as read for write-before-read enforcement
        self.recently_read.add(absolute_path)

        # Format with line numbers
        output_lines = [
            f"{i + start + 1:6d}\t{line}"
            for i, line in enumerate(selected)
        ]
        output = "\n".join(output_lines)

        # Truncate if too long
        MAX_LINES = 2000
        if len(output_lines) > MAX_LINES:
            output_lines = output_lines[:MAX_LINES]
            output = "\n".join(output_lines)
            output += f"\n\n... (truncated at {MAX_LINES} lines, {total_lines} total)"

        return AgentToolResult(
            content=[{"type": "text", "text": output or "(empty file)"}],
        )

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self._cwd, path))


def create_read_tool(cwd: str) -> ReadTool:
    return ReadTool(cwd)
