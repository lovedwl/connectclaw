"""
Feishu (Lark) channel via lark-oapi SDK Channel module.

Uses lark_oapi.channel.FeishuChannel for WebSocket connection,
message receiving, dedup, and sending — no manual wire-up.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from lark_oapi.channel import FeishuChannel as SdkChannel  # noqa: F401 — must import before asyncio loop starts

from connectclaw.logging import get_logger

from .base import Channel, OnMessageCallback

logger = get_logger(__name__)


@dataclass
class FeishuConfig:
    app_id: str = ""
    app_secret: str = ""


@dataclass
class AuthRequest:
    request_id: str
    conversation_key: str
    command: str
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    approved: bool = False


class FeishuChannel(Channel):
    """Feishu channel backed by lark_oapi.channel.FeishuChannel."""

    def __init__(self, config: FeishuConfig):
        self._config = config
        self._sdk: Any = None
        self._running = False
        self._stop_event: asyncio.Event = asyncio.Event()
        self._auth_requests: dict[str, AuthRequest] = {}
        self._auth_results: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()

    # ── Start / Stop ────────────────────────────────────────

    async def start(self, on_message: OnMessageCallback) -> None:
        """Connect via WebSocket using SDK Channel (mirrors doc echo_bot)."""
        sdk = SdkChannel(
            app_id=self._config.app_id,
            app_secret=self._config.app_secret,
        )
        self._sdk = sdk
        channel = self

        async def on_msg(msg) -> None:
            """Handle inbound message from SDK Channel."""
            chat_id = msg.chat_id
            text = (msg.content_text or "").strip()
            if not text:
                return

            logger.info("[%s] %s", chat_id[:8], text[:100])

            try:
                response = await on_message(chat_id, text)
                if response:
                    # Stream response via CardKit for live typing effect
                    await channel._stream_text(chat_id, response)
            except Exception as e:
                logger.error("Message handler error: %s", e)

        sdk.on("message", on_msg)

        async def on_reject(reject) -> None:
            logger.debug("Message rejected: reason=%s", getattr(reject, 'reason', reject))

        sdk.on("reject", on_reject)

        async def on_error(error) -> None:
            logger.error("Channel error: %s", error)

        sdk.on("error", on_error)

        logger.info("Feishu WebSocket connecting...")
        await sdk.connect_until_ready(timeout=30)

        # Keep main event loop alive while WS runs in daemon background thread.
        self._running = True
        self._stop_event.clear()
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Channel keep-alive cancelled, disconnecting...")
        finally:
            self._running = False

    async def close(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._sdk is not None:
            try:
                await self._sdk.disconnect()
            except Exception as e:
                logger.debug("Disconnect error (non-critical): %s", e)
            self._sdk = None

    # ── Sending ─────────────────────────────────────────────

    async def _stream_text(self, conversation_key: str, text: str) -> None:
        """Stream text via CardKit markdown streaming for live typing effect.

        Falls back to plain text send if streaming fails.
        """
        if self._sdk is None:
            logger.error("_stream_text: not connected")
            return

        # Split into paragraphs for natural-feeling streaming
        chunks = _split_for_streaming(text)

        async def producer(stream):
            for chunk in chunks:
                await stream.append(chunk)

        try:
            result = await self._sdk.stream(conversation_key, {"markdown": producer})
            if not result.ok:
                logger.debug("stream failed, falling back to text: %s", result.error)
                await self.send_message(conversation_key, text)
        except Exception as e:
            logger.debug("stream exception, falling back to text: %s", e)
            await self.send_message(conversation_key, text)

    async def send_message(self, conversation_key: str, text: str) -> str:
        """Send text via SDK Channel.send()."""
        if self._sdk is None:
            logger.error("send_message: not connected")
            return ""
        result = await self._sdk.send(conversation_key, {"text": text})
        if result.ok:
            return result.message_id or ""
        logger.error("send_message failed: %s", result.error)
        return ""

    async def send_card(self, conversation_key: str, card: dict) -> str:
        """Send interactive card via SDK Channel.send()."""
        if self._sdk is None:
            logger.error("send_card: not connected")
            return ""
        result = await self._sdk.send(conversation_key, {"card": card})
        if result.ok:
            return result.message_id or ""
        logger.error("send_card failed: %s", result.error)
        return ""

    async def send_thinking_indicator(self, conversation_key: str) -> None:
        pass  # Not supported by Feishu bot API

    async def send_error(self, conversation_key: str, error: str) -> str:
        return await self.send_message(conversation_key, f"Error: {error[:500]}")

    # ── Authorization Cards ─────────────────────────────────

    async def request_bash_authorization(
        self, conversation_key: str, command: str, timeout: float = 60.0
    ) -> bool:
        return await self._request_auth(
            conversation_key=conversation_key,
            title="Bash Authorization",
            template="warning",
            command=command,
            description="The assistant wants to execute:",
            timeout=timeout,
        )

    async def request_network_authorization(
        self, conversation_key: str, command: str, timeout: float = 60.0
    ) -> bool:
        return await self._request_auth(
            conversation_key=conversation_key,
            title="Network Access",
            template="info",
            command=command,
            description="The assistant needs network access for:",
            timeout=timeout,
        )

    async def request_unsandboxed_authorization(
        self, conversation_key: str, command: str, timeout: float = 60.0
    ) -> bool:
        return await self._request_auth(
            conversation_key=conversation_key,
            title="Sandbox Escape",
            template="danger",
            command=command,
            description="The assistant needs to run outside the sandbox:",
            timeout=timeout,
        )

    async def _request_auth(
        self, *, conversation_key: str, title: str, template: str,
        command: str, description: str, timeout: float = 60.0,
    ) -> bool:
        import uuid

        rid = str(uuid.uuid4())[:8]
        auth = AuthRequest(request_id=rid, conversation_key=conversation_key, command=command)
        self._auth_requests[rid] = auth

        card = _build_card(command, rid, title, description, template)
        await self.send_card(conversation_key, card)

        try:
            async with asyncio.timeout(timeout):
                while not auth.resolved:
                    try:
                        r, approved = await asyncio.wait_for(
                            self._auth_results.get(), timeout=1.0
                        )
                        if r == rid:
                            auth.resolved = True
                            auth.approved = approved
                    except asyncio.TimeoutError:
                        continue
        except TimeoutError:
            auth.resolved = True
            auth.approved = False
            await self.send_message(
                conversation_key,
                f"Timed out. Command NOT executed: `{command}`",
            )

        self._auth_requests.pop(rid, None)
        if not auth.approved:
            await self.send_message(conversation_key, f"Denied: `{command}`")
        return auth.approved

    def handle_card_action(self, event: dict) -> None:
        """Handle card button callback (for webhook mode)."""
        action = event.get("event", {}).get("action", {})
        try:
            value = json.loads(action.get("value", "{}"))
        except json.JSONDecodeError:
            return
        rid = value.get("request_id", "")
        approved = value.get("action") == "approve"
        if rid and rid in self._auth_requests:
            self._auth_requests[rid].resolved = True
            self._auth_requests[rid].approved = approved
            self._auth_results.put_nowait((rid, approved))


def _split_for_streaming(text: str) -> list[str]:
    """Split text into chunks for natural CardKit streaming."""
    if not text:
        return [""]

    # Split on double-newline (paragraph boundaries)
    import re
    paragraphs = re.split(r'\n\n+', text)
    chunks: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Add paragraph separator between chunks
        if chunks:
            chunks.append("\n\n")
        # Break long paragraphs on sentence boundaries
        if len(para) > 80:
            sentences = re.split(r'(?<=[.!?。！？])\s+', para)
            for i, s in enumerate(sentences):
                if s:
                    chunks.append(s + (" " if i < len(sentences) - 1 else ""))
        else:
            chunks.append(para)
    return chunks if chunks else [text]


def _build_card(
    command: str, request_id: str, title: str, description: str, template: str
) -> dict:
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"{description}\n\n**Command:**\n```\n{command}\n```\n\nAllow?",
                },
            },
            {"tag": "hr"},
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Approve"},
                        "type": "primary",
                        "value": json.dumps({
                            "action": "approve",
                            "request_id": request_id,
                        }),
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Deny"},
                        "type": "danger",
                        "value": json.dumps({
                            "action": "deny",
                            "request_id": request_id,
                        }),
                    },
                ],
            },
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"Request: {request_id} | Expires 60s",
                    }
                ],
            },
        ],
    }
