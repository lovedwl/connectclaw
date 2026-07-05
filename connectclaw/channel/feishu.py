"""
Feishu (Lark) channel via lark-oapi SDK Channel module.

Uses lark_oapi.channel.FeishuChannel for WebSocket connection,
message receiving, dedup, and sending — no manual wire-up.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
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
    message_id: str = ""  # card message_id for in-place update
    created_at: float = field(default_factory=time.time)
    approved: bool = False
    event: asyncio.Event = field(default_factory=asyncio.Event)


class FeishuChannel(Channel):
    """Feishu channel backed by lark_oapi.channel.FeishuChannel."""

    def __init__(self, config: FeishuConfig):
        self._config = config
        self._sdk: Any = None
        self._running = False
        self._stop_event: asyncio.Event = asyncio.Event()
        self._auth_requests: dict[str, AuthRequest] = {}

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
            """Handle inbound message from SDK Channel.

            The actual processing is scheduled as a separate asyncio task so
            the SDK's ChatPipeline serial queue is released immediately.
            Slash commands skip the live card (no thinking panel).
            """
            chat_id = msg.chat_id
            text = (msg.content_text or "").strip()
            if not text:
                return

            logger.info("[%s] %s", chat_id[:8], text[:100])

            # Commands don't need the live thinking card
            is_cmd = text.startswith("/")

            async def _process() -> None:
                try:
                    callbacks = None if is_cmd else channel.create_live_card(chat_id)
                    response = await on_message(chat_id, text, callbacks)
                    if response:
                        await channel._stream_text(chat_id, response)
                except Exception as e:
                    logger.error("Message handler error: %s", e)

            asyncio.create_task(_process())

        sdk.on("message", on_msg)

        async def on_reject(reject) -> None:
            logger.debug("Message rejected: reason=%s", getattr(reject, 'reason', reject))

        sdk.on("reject", on_reject)

        async def on_error(error) -> None:
            logger.error("Channel error: %s", error)

        sdk.on("error", on_error)

        # Register card action handler (for non-auth cards; auth cards are
        # intercepted earlier via monkey-patch on _on_p2_card_action_trigger)
        sdk.on("cardAction", channel.handle_card_action)

        logger.info("Feishu WebSocket connecting...")
        await sdk.connect_until_ready(timeout=30)

        # Keep main event loop alive while WS runs in daemon background thread.
        self._running = True
        self._stop_event.clear()
        try:
            while self._running and not self._stop_event.is_set():
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("Channel keep-alive cancelled, disconnecting...")
        finally:
            self._running = False

    def close_safe(self) -> None:
        """Signal shutdown without blocking (for signal handlers)."""
        self._running = False
        self._stop_event.set()

    async def close(self) -> None:
        self.close_safe()
        if self._sdk is not None:
            try:
                await asyncio.wait_for(self._sdk.disconnect(), timeout=3)
            except asyncio.TimeoutError:
                logger.debug("Disconnect timed out, forcing stop")
            except Exception as e:
                logger.debug("Disconnect error (non-critical): %s", e)
            self._sdk = None

    # ── Sending ─────────────────────────────────────────────

    async def _stream_text(self, conversation_key: str, text: str) -> None:
        """Send response as markdown for full formatting (tables, bold, etc)."""
        if self._sdk is None:
            logger.error("_stream_text: not connected")
            return
        result = await self._sdk.send(conversation_key, {"markdown": text})
        if not result.ok:
            logger.debug("markdown send failed, falling back to text: %s", result.error)
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

    # ── Live Card (Phase 1 thinking + Phase 2 streaming) ─────

    def create_live_card(self, chat_id: str) -> dict[str, Any]:
        """Create callbacks for the agent loop. Phase 1: regular card with
        collapsible thinking sections (update_card, throttled). Phase 2: streaming text."""
        sdk = self._sdk
        if sdk is None:
            return {}

        state = {
            "chat_id": chat_id,
            "message_id": None,
            "history": "",
            "thinking_buf": "",
            "thinking_start": 0.0,
            "text_started": False,
            "last_flush": 0.0,
        }
        lock = asyncio.Lock()

        async def _flush() -> None:
            """Send or update the thinking card with collapsible sections."""
            async with lock:
                sections = _parse_history(state["history"])
                if state["thinking_buf"]:
                    elapsed = time.time() - state["thinking_start"]
                    sections.append({
                        "title": f"💭 思考中... ({elapsed:.1f}s)",
                        "content": state["thinking_buf"],
                        "expanded": True,
                    })
                card = _build_sections_card(sections)
                if state["message_id"] is None:
                    result = await sdk.send(chat_id, {"card": card})
                    if result.ok and result.message_id:
                        state["message_id"] = result.message_id
                else:
                    try:
                        await sdk.update_card(state["message_id"], card)
                    except Exception as e:
                        logger.debug("update_card failed, re-sending: %s", e)
                        state["message_id"] = None
                        await sdk.send(chat_id, {"card": card})  # fallback: send new

        # Fire card immediately
        asyncio.create_task(_flush())

        # ── Callbacks ──────────────────────────────────────

        async def on_thinking_delta(text: str) -> None:
            now = time.time()
            async with lock:
                if not state["thinking_buf"]:
                    state["thinking_start"] = now
                state["thinking_buf"] += text
                # Throttle: max 1 update per second for thinking progress
                if now - state["last_flush"] < 1.0:
                    return
                state["last_flush"] = now
            await _flush()

        async def on_thinking_done(elapsed: float) -> None:
            async with lock:
                if not state["thinking_buf"]:
                    return
                state["history"] += f"\n\n💭 已思考 {elapsed:.1f}s\n\n{state['thinking_buf']}\n"
                state["thinking_buf"] = ""
            await _flush()

        # Map tool_call_id → short label for result matching
        _tool_labels: dict[str, str] = {}

        async def on_tool_call(name: str, args: dict, tc_id: str = "") -> None:
            async with lock:
                if state["thinking_buf"]:
                    elapsed = time.time() - state["thinking_start"]
                    state["history"] += f"\n\n💭 已思考 {elapsed:.1f}s\n\n{state['thinking_buf']}\n"
                    state["thinking_buf"] = ""
                # Embed a unique marker so results can be matched even
                # when multiple tools share the same name.
                label = f"🔧 **{name}** `[{tc_id[:8] if tc_id else '?'}]`"
                state["history"] += f"\n\n{label}\n```json\n{json.dumps(args, ensure_ascii=False, indent=2)}\n```\n"
                if tc_id:
                    _tool_labels[tc_id] = label
            await _flush()

        async def on_tool_result(name: str, is_error: bool, result_text: str = "",
                                 tc_id: str = "") -> None:
            async with lock:
                status = "✅" if not is_error else "❌"
                old = _tool_labels.pop(tc_id, f"🔧 **{name}**") if tc_id else f"🔧 **{name}**"
                if name == "read" and result_text:
                    lines = result_text.count("\n") + 1
                    summary = f"读取 {lines} 行"
                elif result_text:
                    summary = result_text[:300]
                else:
                    summary = ""
                new = f"{old} {status}"
                parts = state["history"].rsplit(old, 1)
                if len(parts) == 2:
                    body = f"{new}\n{summary}" if summary else new
                    state["history"] = parts[0] + body + parts[1]
            await _flush()

        async def on_text_delta(text: str) -> None:
            needs_flush = False
            async with lock:
                if not state["text_started"]:
                    state["text_started"] = True
                    if state["thinking_buf"]:
                        elapsed = time.time() - state["thinking_start"]
                        state["history"] += f"\n\n💭 已思考 {elapsed:.1f}s\n\n{state['thinking_buf']}\n"
                        state["thinking_buf"] = ""
                    needs_flush = True
            if needs_flush:
                await _flush()

        async def on_text_done() -> None:
            pass

        return {
            "on_thinking_delta": on_thinking_delta,
            "on_thinking_done": on_thinking_done,
            "on_tool_call": on_tool_call,
            "on_tool_result": on_tool_result,
            "on_text_delta": on_text_delta,
            "on_text_done": on_text_done,
        }

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
        rid = str(uuid.uuid4())[:8]
        auth = AuthRequest(request_id=rid, conversation_key=conversation_key, command=command)
        self._auth_requests[rid] = auth

        card = _build_card(command, rid, title, description, template)
        msg_id = await self.send_card(conversation_key, card)
        auth.message_id = msg_id
        logger.info("[AUTH] sent card for %s: message_id=%s command=%s",
                    rid, msg_id, command[:80])

        try:
            async with asyncio.timeout(timeout):
                await auth.event.wait()
        except TimeoutError:
            auth.approved = False
            logger.info("[AUTH] %s timed out after %ss", rid, timeout)
            if msg_id:
                await self._update_auth_card(msg_id, command, rid,
                                             approved=False, timed_out=True)

        self._auth_requests.pop(rid, None)
        if msg_id:
            await self._update_auth_card(msg_id, command, rid,
                                         approved=auth.approved, timed_out=False)
        return auth.approved

    async def _update_auth_card(
        self, message_id: str,
        command: str, request_id: str,
        approved: bool, timed_out: bool,
    ) -> None:
        """Replace the auth card with a resolved state (no buttons)."""
        if timed_out:
            header_color = "grey"
            status_text = "⏰ Timed out"
            status_detail = f"Authorization request expired.\n\n**Command:**\n```\n{command}\n```"
        elif approved:
            header_color = "green"
            status_text = "✅ Approved"
            status_detail = f"Command will be executed.\n\n**Command:**\n```\n{command}\n```"
        else:
            header_color = "red"
            status_text = "❌ Denied"
            status_detail = f"Command will NOT be executed.\n\n**Command:**\n```\n{command}\n```"

        resolved_card = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": status_text},
                "template": header_color,
            },
            "body": {
                "elements": [
                    {"tag": "markdown", "content": status_detail},
                    {"tag": "markdown",
                     "content": f"<font color='grey'>Request: {request_id}</font>"},
                ],
            },
        }
        try:
            await self._sdk.update_card(message_id, resolved_card)
        except Exception as e:
            logger.debug("update_card for auth result failed: %s", e)

    def handle_card_action(self, event) -> None:
        """Handle card button callback (WebSocket mode).

        Receives a ``CardActionEvent`` dataclass from the SDK Channel.
        Extracts ``request_id`` and ``action`` from the button's JSON value
        and resolves the pending :class:`AuthRequest`.
        """
        action_value = event.action.value
        logger.info("[AUTH] cardAction received: raw_value=%s type=%s tag=%s",
                    repr(action_value), type(action_value).__name__, event.action.tag)

        # SDK normalizes: JSON string → dict, but keep the string fallback
        if isinstance(action_value, str):
            try:
                action_value = json.loads(action_value)
            except json.JSONDecodeError:
                logger.warning("[AUTH] cardAction value is not valid JSON: %s", action_value[:200])
                return
        if not isinstance(action_value, dict):
            logger.warning("[AUTH] cardAction value is not a dict: %s", type(action_value).__name__)
            return

        rid = action_value.get("request_id", "")
        approved = action_value.get("action") == "approve"
        logger.info("[AUTH] request_id=%s approved=%s pending_requests=%s",
                    rid, approved, list(self._auth_requests.keys()))

        if rid and rid in self._auth_requests:
            self._auth_requests[rid].approved = approved
            self._auth_requests[rid].event.set()
            logger.info("[AUTH] resolved request %s: approved=%s", rid, approved)
        else:
            logger.warning("[AUTH] unknown or stale request_id=%s (pending: %s)",
                           rid, list(self._auth_requests.keys())[:5])


def _split_for_streaming(text: str) -> list[str]:
    """Split text at punctuation boundaries for smooth CardKit streaming.

    Each chunk is a natural language clause — Chinese flows character by
    character, English keeps words together. CardKit's ~2 chars/50ms makes
    this feel like real-time typing.
    """
    if not text:
        return [""]

    # Split at sentence/clause boundaries.
    # '.' only when followed by space/newline/end (English period),
    # not when part of filenames/numbers/URLs (main.py, 3.14).
    parts = re.split(r'(?<=[。！？；：!?\n])\s*|(?<=\.)(?=\s+|$)\s*', text)
    return [p for p in parts if p]


def _parse_history(history: str) -> list[dict]:
    """Parse history string into [{title, content, expanded}, ...]."""
    sections = []
    parts = re.split(r'\n\n(?=💭 已思考|🔧 )', history.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if len(body) > 1500:
            body = body[:1500] + "..."
        sections.append({
            "title": title,
            "content": body,
            "expanded": False,
        })
    # Keep last 5, merge older
    if len(sections) > 3:
        older = sections[:-3]
        sections = sections[-3:]
        older_content = "\n\n".join(
            f"**{s['title']}**\n{s['content']}" for s in older
        )
        sections.insert(0, {
            "title": f"📋 更早的 {len(older)} 个步骤",
            "content": older_content,
            "expanded": False,
        })
    return sections


def _build_sections_card(sections: list[dict]) -> dict:
    """Build a card JSON with collapsible panel sections."""
    elements = []
    for sec in sections:
        elements.append({
            "tag": "collapsible_panel",
            "expanded": sec.get("expanded", False),
            "background_color": "grey",
            "padding": "8px 8px 8px 8px",
            "margin": "4px 0px 4px 0px",
            "border": {"color": "grey", "corner_radius": "6px"},
            "header": {
                "title": {"tag": "plain_text", "content": sec["title"]},
                "background_color": "grey",
            },
            "elements": [
                {"tag": "markdown", "content": sec.get("content", "") or " "}
            ],
        })
    if not elements:
        elements.append({"tag": "markdown", "content": "💭 思考中..."})
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "body": {"elements": elements},
    }


def _build_card(
    command: str, request_id: str, title: str, description: str, template: str
) -> dict:
    # CardKit v2 notes (from SDK card/builder.py):
    # - `action` container tag is gone — buttons go directly in body.elements
    # - `note` tag is gone — use grey markdown instead
    # - Multiple buttons in one row → column_set
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        },
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": f"{description}\n\n**Command:**\n```\n{command}\n```\n\nAllow?",
                },
                {"tag": "hr"},
                {
                    "tag": "column_set",
                    "columns": [
                        {
                            "tag": "column",
                            "elements": [
                                {
                                    "tag": "button",
                                    "text": {"tag": "plain_text", "content": "✅ Approve"},
                                    "type": "primary",
                                    "value": {
                                        "action": "approve",
                                        "request_id": request_id,
                                    },
                                },
                            ],
                        },
                        {
                            "tag": "column",
                            "elements": [
                                {
                                    "tag": "button",
                                    "text": {"tag": "plain_text", "content": "❌ Deny"},
                                    "type": "danger",
                                    "value": {
                                        "action": "deny",
                                        "request_id": request_id,
                                    },
                                },
                            ],
                        },
                    ],
                },
                {
                    "tag": "markdown",
                    "content": f"<font color='grey'>Request: {request_id} | Expires 60s</font>",
                },
            ],
        },
    }
