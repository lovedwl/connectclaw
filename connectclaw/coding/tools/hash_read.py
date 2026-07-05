"""Hashline read tool — reads files with LINE#HASH:content anchors."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import aiofiles

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.hashline.format import format_hashline_region
from connectclaw.hashline.snapshot import remember_read_snapshot
from connectclaw.hashline.guard import clear_applied_payload
from connectclaw.hashline.diff_util import normalize_to_lf, strip_bom

# ─── Truncation ─────────────────────────────────────────────────────────────

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 256 * 1024  # 256 KiB


def _format_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


# ─── Tool Description ───────────────────────────────────────────────────────

HASH_READ_DESCRIPTION = f"""Read a text file with hash-anchored line output.

Every line returns as LINE#HASH:content — copy those anchors verbatim into
hash_edit. They are the only way hash_edit addresses lines.

Page large files with offset (1-based line) and limit.
Default cap: {DEFAULT_MAX_LINES} lines or {_format_size(DEFAULT_MAX_BYTES)}.
Truncated output ends with the exact offset to continue from.

Set raw: true to return plain file content without LINE#HASH prefixes.
Offset, limit, and truncation notices still apply.
Use raw mode to save tokens when you do not plan to edit this file."""


class HashReadTool(AgentTool):
    name = "hash_read"
    label = "Hash Read"
    description = HASH_READ_DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative or absolute)",
            },
            "offset": {
                "type": "integer",
                "minimum": 1,
                "description": "Line number to start reading from (1-indexed)",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum number of lines to read",
            },
            "raw": {
                "type": "boolean",
                "description": "Return plain text without LINE#HASH anchors. Saves tokens when you do not plan to edit this file.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, cwd: str):
        self._cwd = cwd

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        file_path = params["path"]
        absolute_path = self._resolve_path(file_path)

        # Check file exists
        if not os.path.isfile(absolute_path):
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error: File not found: {file_path}"}],
            )

        # Check is not a directory
        if os.path.isdir(absolute_path):
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error: Path is a directory: {file_path}"}],
            )

        # Read file
        try:
            async with aiofiles.open(absolute_path, "r") as f:
                content = await f.read()
        except UnicodeDecodeError:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": f"Error: File is binary or not UTF-8: {file_path}. Use the read tool for binary inspection.",
                }],
            )
        except Exception as e:
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error reading file: {e}"}],
            )

        # Strip BOM, normalize line endings
        bom, text = strip_bom(content)
        normalized = normalize_to_lf(text)

        # Parse into lines (visible only — no trailing sentinel)
        if normalized.endswith("\n"):
            all_lines = normalized[:-1].split("\n")
        else:
            all_lines = normalized.split("\n")

        total_lines = len(all_lines)
        raw_mode = params.get("raw", False)

        # Apply offset and limit
        offset = params.get("offset", 1) or 1
        if offset < 1:
            offset = 1

        limit = params.get("limit")

        if total_lines == 0:
            msg = (
                "File is empty. Use hash_edit with prepend or append "
                "and omit pos to insert content."
            )
            if offset > 1:
                msg = (
                    f"Offset {offset} is beyond end of file (0 lines total). "
                    f"The file is empty. Use hash_edit with prepend or append "
                    f"and omit pos to insert content."
                )
            return AgentToolResult(content=[{"type": "text", "text": msg}])

        if offset > total_lines:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"Offset {offset} is beyond end of file "
                        f"({total_lines} lines total). "
                        f"Use offset=1 to read from the start, "
                        f"or offset={total_lines} to read the last line."
                    ),
                }]
            )

        start_idx = offset - 1
        end_idx = min(start_idx + limit, total_lines) if limit else total_lines

        # Format output
        if raw_mode:
            selected = all_lines[start_idx:end_idx]
            output = "\n".join(selected)
        else:
            output = format_hashline_region(all_lines, start_idx + 1, end_idx)

        # Truncation check
        MAX_BYTES = DEFAULT_MAX_BYTES
        output_bytes = output.encode("utf-8")
        truncated = False
        if len(output_bytes) > MAX_BYTES:
            # Truncate by lines to stay under byte limit
            truncated = True
            if not raw_mode:
                # For hashline mode, check first line
                first_line = all_lines[start_idx]
                first_line_bytes = (
                    format_hashline_region([first_line], start_idx + 1, start_idx + 1)
                    .encode("utf-8")
                )
                if len(first_line_bytes) > MAX_BYTES:
                    return AgentToolResult(
                        content=[{
                            "type": "text",
                            "text": (
                                f"[Line {start_idx + 1} exceeds "
                                f"{_format_size(MAX_BYTES)}. "
                                f"Hashline output requires full lines; "
                                f"cannot compute hashes for a truncated preview.]"
                            ),
                        }]
                    )
            # Truncate by reducing end_idx
            while len(output.encode("utf-8")) > MAX_BYTES and end_idx > start_idx + 1:
                end_idx -= 1
                if raw_mode:
                    selected = all_lines[start_idx:end_idx]
                    output = "\n".join(selected)
                else:
                    output = format_hashline_region(all_lines, start_idx + 1, end_idx)

        # Continuation notice
        if truncated:
            if raw_mode:
                output += (
                    f"\n\n[Showing lines {offset}-{end_idx} of {total_lines}"
                    f" ({_format_size(MAX_BYTES)} limit)."
                    f" Use offset={end_idx + 1} to continue.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {offset}-{end_idx} of {total_lines}"
                    f" ({_format_size(MAX_BYTES)} limit)."
                    f" Use offset={end_idx + 1} to continue.]"
                )
        elif end_idx < total_lines:
            output += (
                f"\n\n[Showing lines {offset}-{end_idx} of {total_lines}."
                f" Use offset={end_idx + 1} to continue.]"
            )

        # Add UTF-8 decode warning
        had_utf8_errors = bom != ""  # simplify: BOM presence is the main signal
        if had_utf8_errors:
            output += "\n\n[Non-UTF-8 bytes shown as U+FFFD; editing rewrites the file as UTF-8.]"

        # Record snapshot for stale-anchor recovery (hashed reads only)
        if not raw_mode:
            remember_read_snapshot(absolute_path, normalized)
            clear_applied_payload(absolute_path)

        return AgentToolResult(
            content=[{"type": "text", "text": output}],
        )

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self._cwd, path))


def create_hash_read_tool(cwd: str) -> HashReadTool:
    return HashReadTool(cwd)
