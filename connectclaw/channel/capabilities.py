"""Channel-backed agent tools — the agent's window into IM channel capabilities.

Channel features (send a file, send an image, later: live cards, media, ...)
belong to the channel layer, NOT to coding/: they vary per IM platform and are
implemented on the concrete channel (FeishuChannel), then exposed to the agent
through a thin, interface-only binding that lives here alongside the Channel
ABC. The agent layer only ever sees `Channel` + this tool — swap the channel
implementation and the tools keep working.

To add a new channel capability (e.g. live_card streaming):
  1. Declare it on `Channel` (channel/base.py).
  2. Implement it on the concrete channel (channel/feishu.py).
  3. (Optional) bind it to the agent here as a new tool.

These tools import only leaf modules (agent.types, logging) — never the agent
runtime — so channel → agent stays acyclic.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.channel.base import Channel
from connectclaw.logging import get_logger

logger = get_logger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}


class SendFileTool(AgentTool):
    name = "send_file"
    label = "send_file"
    description = (
        "把本地文件/图片以聊天消息形式发给当前用户。图片（png/jpg/webp/gif/bmp/svg）"
        "默认以图片消息发送，其他类型以文件附件发送；可用 as_image 强制指定。"
        "适用于用户索要你生成、整理或保存的文件（报告、表格、图、代码清单等）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要发送的本地文件绝对路径",
            },
            "as_image": {
                "type": "boolean",
                "description": "可选：true 强制按图片发送，false 强制按文件发送，缺省按扩展名自动判断",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        *,
        channel: Channel | None = None,
        channel_provider: Callable[[], Channel | None] | None = None,
        chat_provider: Callable[[], str],
    ):
        """Bind a concrete channel either directly or via a provider (provider
        wins — coding_agent injects the live channel + active chat per turn)."""
        self._channel = channel
        self._channel_provider = channel_provider or (lambda: channel)
        self._chat_provider = chat_provider

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        path = (params.get("path") or "").strip()
        if not path:
            return self._err("需要 path 参数（要发送的本地文件绝对路径）")
        if not os.path.isfile(path):
            return self._err(f"文件不存在：{path}")

        channel = self._channel_provider()
        chat = self._chat_provider()
        if channel is None or not chat:
            return self._err("当前没有可用的会话，无法发送文件")

        as_image = params.get("as_image")
        if as_image is None:
            as_image = os.path.splitext(path)[1].lower() in _IMAGE_EXTS

        try:
            if as_image:
                label = "图片"
                mid = await channel.send_image(chat, path)
            else:
                label = "文件"
                mid = await channel.send_file(chat, path)
        except Exception as e:  # noqa: BLE001
            logger.warning("send_file failed: %s", e)
            return self._err(f"发送{label}失败：{e}")

        if not mid:
            return self._err(f"发送{label}失败（无 message_id，请查看服务日志）")

        logger.info("sent %s: %s", label, os.path.basename(path))
        return AgentToolResult(
            content=[{"type": "text", "text": f"✅ 已发送{label}：{os.path.basename(path)}"}],
            details={"message_id": mid, "kind": label},
        )

    @staticmethod
    def _err(msg: str) -> AgentToolResult:
        return AgentToolResult(content=[{"type": "text", "text": msg}], details={"is_error": True})


def create_send_file_tool(
    *,
    channel: Channel | None = None,
    channel_provider: Callable[[], Channel | None] | None = None,
    chat_provider: Callable[[], str],
) -> SendFileTool:
    return SendFileTool(
        channel=channel,
        channel_provider=channel_provider,
        chat_provider=chat_provider,
    )
