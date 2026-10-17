"""Write tool — writes file content with read-before-write enforcement."""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import aiofiles

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.security import (
    is_protected,
    path_within_cwd,
    resolve_tool_path,
)


class WriteTool(AgentTool):
    name = "write"
    label = "write"
    description = (
        "Write content to a file. Only files within the working directory can "
        "be written. If the file already exists, read it first (read or "
        "hash_read). The write is atomic."
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
        file_path = params.get("file_path")
        content = params.get("content")
        if not isinstance(file_path, str) or not file_path.strip():
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        "错误：缺少必填参数 file_path。若本调用带有大体积 payload，"
                        "参数可能已在传输中被截断 —— 请重试本调用，或把内容拆小。"
                    ),
                }],
                details={"is_error": True},
            )
        if not isinstance(content, str):
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": "错误：缺少必填参数 content（文件内容），或其在传输中被截断。请重试本调用。",
                }],
                details={"is_error": True},
            )
        absolute_path = self._resolve_path(file_path)

        # Agent must never modify the escape-hatch security files (operator
        # whitelist in config.toml, model registry). Refuse outright.
        if is_protected(absolute_path):
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"⛔ 受保护文件（逃生配置，agent 不可修改）：{absolute_path}"
                    ),
                }],
                details={"is_error": True},
            )

        # Only allow writes within cwd — bash runs in sandbox which can't
        # see files outside cwd, so write must match.
        # realpath 容器检查：symlink 指向 cwd 外、或绝对路径中的 .. 都拒绝。
        if not path_within_cwd(self._cwd, absolute_path):
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"错误：只能写入工作目录（"
                        f"{self._cwd}）内的文件。'{file_path}' 在目录之外。"
                        f"请使用 cwd 内的 /tmp 路径，或写入项目目录。"
                    ),
                }],
                details={"is_error": True},
            )

        # Enforce read-before-write for existing files
        if os.path.exists(absolute_path):
            recently_read: set[str] = getattr(self._read_tool, "recently_read", set())
            if absolute_path not in recently_read:
                return AgentToolResult(
                    content=[{
                        "type": "text",
                        "text": (
                            f"错误：文件 '{file_path}' 已存在但尚未读取。"
                            f"写入前必须先用 read 或 hash_read 工具读取该文件。"
                        ),
                    }],
                    details={"is_error": True},
                )

        # Atomic write: write to temp file, then rename
        try:
            os.makedirs(os.path.dirname(absolute_path) or ".", exist_ok=True)

            # Write to temp file in same directory (ensures same filesystem for rename)
            dir_name = os.path.dirname(absolute_path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".connectclaw_write_")
            try:
                async with aiofiles.open(fd, "w", closefd=False, newline="") as f:
                    await f.write(content)
            finally:
                os.close(fd)

            # os.replace 迁移的是临时文件的 mode（mkstemp = 0600）；
            # 覆盖已有文件时必须还原其权限位（H14 同款，可执行位会静默丢失）。
            try:
                st = os.stat(absolute_path)
                os.chmod(tmp_path, st.st_mode & 0o7777)
            except FileNotFoundError:
                pass

            # Atomic rename
            os.replace(tmp_path, absolute_path)

            size = len(content.encode("utf-8"))
            lines = content.count("\n") + 1
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": f"成功写入 {size} 字节（{lines} 行）到 {file_path}",
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
                content=[{"type": "text", "text": f"写入文件错误：{e}"}],
                details={"is_error": True},
            )

    def _resolve_path(self, path: str) -> str:
        # normpath 绝对路径：/proj/../proj2/f 不能以 startswith("/proj/") 放行（M24）
        return resolve_tool_path(self._cwd, path)


def create_write_tool(cwd: str, read_tool: object) -> WriteTool:
    return WriteTool(cwd, read_tool)
