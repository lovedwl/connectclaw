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
from connectclaw.security import (
    is_protected,
    path_within_cwd,
    resolve_tool_path,
)

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

    def __init__(self, cwd: str, read_tool: object = None, allow_outside_cwd: bool = False):
        self._cwd = cwd
        self._read_tool = read_tool
        self._allow_outside_cwd = allow_outside_cwd
        if allow_outside_cwd:
            self.description += (
                "\nFiles outside the working directory are readable "
                "(full host access is enabled)."
            )

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        file_path = params["path"]
        absolute_path = self._resolve_path(file_path)

        # 2026-10 (H2)：与 read/write 同一契约——默认只读 cwd 内文件（sandbox
        # 为 none 时解除），逃生配置直接拒绝。此前命中保护只是加警告后照样
        # 返回完整内容，config.toml / models.toml 里的密钥仍会被贴回聊天。
        if not self._allow_outside_cwd and not path_within_cwd(self._cwd, absolute_path):
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"错误：只能读取工作目录（{self._cwd}）内的文件。"
                        f"'{file_path}' 在目录之外。"
                    ),
                }],
                details={"is_error": True},
            )
        if is_protected(absolute_path):
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"⛔ 受保护文件（逃生配置，agent 不可读取）：{absolute_path}"
                    ),
                }],
                details={"is_error": True},
            )

        # Check file exists
        if not os.path.isfile(absolute_path):
            return AgentToolResult(
                content=[{"type": "text", "text": f"错误：文件不存在：{file_path}"}],
                details={"is_error": True},
            )

        # M27：超大文件整读会打爆内存——超限拒绝
        _MAX_READ_BYTES = 32 * 1024 * 1024
        try:
            if os.path.getsize(absolute_path) > _MAX_READ_BYTES:
                return AgentToolResult(
                    content=[{
                        "type": "text",
                        "text": (
                            f"错误：文件过大（> {_MAX_READ_BYTES // (1024 * 1024)}MB），"
                            f"hashline 编辑仅适用于常规文本文件。"
                        ),
                    }],
                    details={"is_error": True},
                )
        except OSError:
            pass

        # Check is not a directory
        if os.path.isdir(absolute_path):
            return AgentToolResult(
                content=[{"type": "text", "text": f"错误：路径是目录：{file_path}"}],
                details={"is_error": True},
            )

        # Read file
        try:
            # newline=""（H12）：避免 universal newlines 折叠 \r\n，
            # 否则锚点/内容与磁盘实际行尾不一致，hash_edit 会整文件改写行尾。
            async with aiofiles.open(absolute_path, "r", newline="") as f:
                content = await f.read()
        except UnicodeDecodeError:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"错误：文件为二进制或非 UTF-8：{file_path}。"
                        f"hashline 仅适用于文本文件；图片用 view_image 查看。"
                    ),
                }],
                details={"is_error": True},
            )
        except Exception as e:
            return AgentToolResult(
                content=[{"type": "text", "text": f"读取文件错误：{e}"}],
                details={"is_error": True},
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
                "文件为空。请使用 hash_edit 的 prepend 或 append "
                "并省略 pos 以插入内容。"
            )
            if offset > 1:
                msg = (
                    f"偏移量 {offset} 超出文件末尾（共 0 行）。"
                    f"文件为空。请使用 hash_edit 的 prepend 或 append "
                    f"并省略 pos 以插入内容。"
                )
            return AgentToolResult(content=[{"type": "text", "text": msg}])

        if offset > total_lines:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"偏移量 {offset} 超出文件末尾"
                        f"（共 {total_lines} 行）。"
                        f"使用 offset=1 从头读取，"
                        f"或 offset={total_lines} 读取最后一行。"
                    ),
                }],
                details={"is_error": True},
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
                                f"[第 {start_idx + 1} 行超出 "
                                f"{_format_size(MAX_BYTES)}。"
                                f"Hashline 输出需要完整行；"
                                f"无法为已截断的预览计算哈希。]"
                            ),
                        }],
                        details={"is_error": True},
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
                    f"\n\n[显示第 {offset}-{end_idx} 行，共 {total_lines} 行"
                    f"（限制 {_format_size(MAX_BYTES)}）。"
                    f" 使用 offset={end_idx + 1} 继续。]"
                )
            else:
                output += (
                    f"\n\n[显示第 {offset}-{end_idx} 行，共 {total_lines} 行"
                    f"（限制 {_format_size(MAX_BYTES)}）。"
                    f" 使用 offset={end_idx + 1} 继续。]"
                )
        elif end_idx < total_lines:
            output += (
                f"\n\n[显示第 {offset}-{end_idx} 行，共 {total_lines} 行。"
                f" 使用 offset={end_idx + 1} 继续。]"
            )

        # Add UTF-8 decode warning
        had_utf8_errors = bom != ""  # simplify: BOM presence is the main signal
        if had_utf8_errors:
            output += "\n\n[非 UTF-8 字节显示为 U+FFFD；编辑会将文件重写为 UTF-8。]"

        # Record snapshot for stale-anchor recovery (hashed reads only)
        if not raw_mode:
            remember_read_snapshot(absolute_path, normalized)
            clear_applied_payload(absolute_path)

        # 2026-10 R2 bridge: a successful hash_read also satisfies the legacy
        # read tool's read-before-write ledger, so hash_read -> write is no
        # longer refused (the two generations kept separate ledgers before).
        if self._read_tool is not None:
            recently = getattr(self._read_tool, "recently_read", None)
            if isinstance(recently, set):
                recently.add(absolute_path)

        # 2026-10 R6 → 2026-10 (H2)：受保护文件改为直接拒绝（原为警告后照读）。

        return AgentToolResult(
            content=[{"type": "text", "text": output}],
        )

    def _resolve_path(self, path: str) -> str:
        return resolve_tool_path(self._cwd, path)


def create_hash_read_tool(
    cwd: str, read_tool: object = None, allow_outside_cwd: bool = False,
) -> HashReadTool:
    return HashReadTool(cwd, read_tool, allow_outside_cwd=allow_outside_cwd)
