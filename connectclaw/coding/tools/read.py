"""Read tool — reads file contents with line numbers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.security import is_protected, path_within_cwd, resolve_tool_path


class ReadTool(AgentTool):
    name = "read"
    label = "read"
    description = (
        "Read the contents of a TEXT file (UTF-8) with line numbers. "
        "Supports offset/limit for ranges. Images and binaries cannot be read: "
        "use view_image to view images, send_file to send files to the user."
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

        # 2026-10 (H2)：read 与 write 同一契约——只读 cwd 内文件，且逃生配置
        # （config.toml / models.toml）直接拒绝，否则模型一句 read ~/.connectclaw/
        # config.toml 就把 app_secret / 所有 provider key 贴回聊天。
        if not path_within_cwd(self._cwd, absolute_path):
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

        # P2（表情包事故）：图片扩展名直接拦截——不等 UTF-8 解码失败才报错，
        # 提示更早、更明确（模型该知道 read 从来不是看图的工具）。
        if os.path.splitext(absolute_path)[1].lower() in {
            ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico", ".tiff",
        }:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"错误：'{file_path}' 是图片，read 只能读文本文件。"
                        f"查看用 view_image，发送给用户用 send_file。"
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

        # M27：超大文件整读会打爆内存——超限拒绝，让模型用 offset/limit
        # 分段读取，或用 bash（head/grep）处理。
        _MAX_READ_BYTES = 64 * 1024 * 1024
        try:
            if os.path.getsize(absolute_path) > _MAX_READ_BYTES:
                return AgentToolResult(
                    content=[{
                        "type": "text",
                        "text": (
                            f"错误：文件过大（> {_MAX_READ_BYTES // (1024 * 1024)}MB），拒绝整读。"
                            f"注意 offset/limit 也无法读取（实现仍需整读），"
                            f"请用 bash（head -c / tail -c / grep）处理。"
                        ),
                    }],
                    details={"is_error": True},
                )
        except OSError:
            pass

        # Read file
        try:
            # newline=""（H12）：保持磁盘真实行尾，与 hash_read/hash_edit 一致。
            async with aiofiles.open(absolute_path, "r", newline="") as f:
                content = await f.read()
        except UnicodeDecodeError:
            # P2（2026-10-16 表情包事故）：读图片/二进制不再返回晦涩的
            # codec 报错——直接告诉模型能力边界与替代路径，避免它病急乱投医。
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"错误：'{file_path}' 是二进制文件，read 只能读文本。"
                        f"若它是图片：查看用 view_image，发送用 send_file。"
                    ),
                }],
                details={"is_error": True},
            )
        except Exception as e:
            return AgentToolResult(
                content=[{"type": "text", "text": f"读取文件错误：{e}"}],
                details={"is_error": True},
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
            output += f"\n\n... （已在 {MAX_LINES} 行处截断，共 {total_lines} 行）"

        return AgentToolResult(
            content=[{"type": "text", "text": output or "（空文件）"}],
        )

    def _resolve_path(self, path: str) -> str:
        return resolve_tool_path(self._cwd, path)


def create_read_tool(cwd: str) -> ReadTool:
    return ReadTool(cwd)
