"""Write tool — writes file content with read-before-write enforcement."""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import aiofiles

from connectclaw.agent.types import AgentTool, AgentToolResult


class WriteTool(AgentTool):
    name = "write"
    label = "write"
    description = (
        "Write content to a file. "
        "If the file already exists, you MUST read it first using the read tool. "
        "The file will be written atomically."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    }

    def __init__(self, cwd: str, read_tool: object):
        self._cwd = cwd
        self._read_tool = read_tool

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        file_path = params["file_path"]
        content = params["content"]
        absolute_path = self._resolve_path(file_path)

        # Enforce read-before-write for existing files
        if os.path.exists(absolute_path):
            recently_read: set[str] = getattr(self._read_tool, "recently_read", set())
            if absolute_path not in recently_read:
                return AgentToolResult(
                    content=[{
                        "type": "text",
                        "text": (
                            f"Error: File '{file_path}' already exists but has not been read. "
                            f"You must use the read tool to read the file first before writing to it."
                        ),
                    }],
                )

        # Atomic write: write to temp file, then rename
        try:
            os.makedirs(os.path.dirname(absolute_path) or ".", exist_ok=True)

            # Write to temp file in same directory (ensures same filesystem for rename)
            dir_name = os.path.dirname(absolute_path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".connectclaw_write_")
            try:
                async with aiofiles.open(fd, "w", closefd=False) as f:
                    await f.write(content)
            finally:
                os.close(fd)

            # Atomic rename
            os.replace(tmp_path, absolute_path)

            size = len(content.encode("utf-8"))
            lines = content.count("\n") + 1
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": f"Successfully wrote {size} bytes ({lines} lines) to {file_path}",
                }],
            )
        except Exception as e:
            # Clean up temp file if it exists
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error writing file: {e}"}],
            )

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self._cwd, path))


def create_write_tool(cwd: str, read_tool: object) -> WriteTool:
    return WriteTool(cwd, read_tool)
