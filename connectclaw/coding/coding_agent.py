"""
CodingAgent — assembles AgentHarness + tools + RAG + safety + sandbox auth.

This is a general-purpose AI assistant. Coding is one capability among many.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from connectclaw.agent.harness.agent_harness import AgentHarness
from connectclaw.logging import get_logger
from connectclaw.agent.harness.compaction import CompactionSettings
from connectclaw.agent.harness.prompt_builder import PromptBuilder
from connectclaw.agent.harness.rag.subsystem import RAGConfig, RAGSubsystem
from connectclaw.agent.harness.session import SessionRepo
from connectclaw.agent.types import AgentTool
from connectclaw.channel.feishu import FeishuChannel
from connectclaw.config import Config
from connectclaw.provider.types import Model

from .tools.bash import BashGuard, create_bash_tool
from .tools.dynamic import load_dynamic_tools
from .tools.image_analyze import create_image_analyze_tool
from .tools.read import create_read_tool
from .tools.task import create_task_tool
from .tools.web_search import create_web_fetch_tool, create_web_search_tool
from .tools.write import create_write_tool

logger = get_logger(__name__)


class CodingAgent:
    """Assembles a general-purpose AI assistant. Coding is one capability among many."""

    def __init__(self, config: Config | None = None, channel: FeishuChannel | None = None):
        if config is None:
            config = Config.load()

        self._config = config
        self._channel = channel

        # Build model
        self._model = Model(
            id=config.llm.model_id,
            name=config.llm.model_id,
            provider="openai-compatible",
            base_url=config.llm.base_url,
            api="openai-compatible",
            reasoning=True,
            context_window=65536,
            max_tokens=8192,
        )

        # Create base tools (always available)
        self._read_tool = create_read_tool(config.agent.cwd)
        self._write_tool = create_write_tool(config.agent.cwd, self._read_tool)
        self._bash_guard = BashGuard()
        self._bash_tool = create_bash_tool(config.agent.cwd, self._bash_guard)
        self._web_search_tool = create_web_search_tool(
            glyph_bin=config.web_search.glyph_bin,
            max_chars=config.web_search.max_chars,
            timeout=config.web_search.timeout,
        )
        self._web_fetch_tool = create_web_fetch_tool(
            glyph_bin=config.web_search.glyph_bin,
            max_chars=config.web_search.max_chars,
            timeout=config.web_search.timeout,
        )
        self._image_tool = create_image_analyze_tool(
            api_key=config.vision.api_key,
            base_url=config.vision.base_url,
            model_id=config.vision.model_id,
            cwd=config.agent.cwd,
        )

        # Dynamic tools directory
        self._tools_dir = os.path.expanduser("~/.connectclaw/tools")
        os.makedirs(self._tools_dir, exist_ok=True)

        # Task tool (needs all_tools, refreshed each turn)
        self._task_tool = create_task_tool(
            self._model,
            api_key=config.llm.api_key or None,
            thinking_level=config.agent.thinking_level,  # type: ignore[arg-type]
            all_tools=[],  # will be updated in _refresh_tools()
        )

        # Current tool set
        self._tools: list[AgentTool] = []

        # Prompt builder — loads template from ~/.connectclaw/prompts/system.md
        self._prompt_builder = PromptBuilder(cwd=config.agent.cwd)

        # RAG subsystem (optional, lazy init)
        self._rag = RAGSubsystem(
            RAGConfig(
                enabled=config.rag.enabled,
                docs_dir=config.rag.docs_dir,
                db_path=config.rag.db_path,
                top_k=config.rag.top_k,
                top_n=config.rag.top_n,
            )
        )

        # Session repository
        sessions_dir = os.path.expanduser(config.session.dir)
        self._session_repo = SessionRepo(sessions_dir)

        # Compaction settings
        self._compaction_settings = CompactionSettings(
            enabled=config.compaction.enabled,
            reserve_tokens=config.compaction.reserve_tokens,
            keep_recent_tokens=config.compaction.keep_recent_tokens,
        )

        # Per-conversation harnesses
        self._conversations: dict[str, AgentHarness] = {}
        # Track running tasks so /stop can cancel them
        self._running_tasks: dict[str, asyncio.Task] = {}

    @property
    def tools(self) -> list[AgentTool]:
        return list(self._tools)

    @property
    def rag(self) -> RAGSubsystem:
        return self._rag

    @property
    def prompt_builder(self) -> PromptBuilder:
        return self._prompt_builder

    # ── Dynamic Tool Refresh ────────────────────────────────

    def _refresh_tools(self) -> list[AgentTool]:
        """Rebuild tool list: base tools + task + dynamic tools.

        Called each turn so agent-created tools appear immediately.
        """
        dynamic = load_dynamic_tools(self._tools_dir, self._config.agent.cwd)

        base = [
            self._read_tool,
            self._write_tool,
            self._bash_tool,
            self._web_search_tool,
            self._web_fetch_tool,
            self._image_tool,
        ]

        # Update task tool's all_tools reference so it can resolve sub-agent tool names
        all_available = base + dynamic
        self._task_tool._all_tools = all_available

        tools = base + [self._task_tool] + dynamic
        self._tools = tools
        return tools

    # ── System Prompt ───────────────────────────────────────

    def build_system_prompt(self, rag_context: str = "") -> str:
        """Build the full system prompt with tools, skills, RAG, and env info."""
        return self._prompt_builder.build(rag_context=rag_context)

    # ── Conversation Management ─────────────────────────────

    async def handle_message(
        self, conversation_key: str, text: str, live_card_callbacks: dict[str, Any] | None = None
    ) -> str | None:
        """Handle an incoming message."""
        # Track this task so /stop can cancel it
        task = asyncio.current_task()
        if task is not None:
            # Cancel any previous task still running for this conversation
            prev = self._running_tasks.get(conversation_key)
            if prev is not None and not prev.done():
                prev.cancel()
            self._running_tasks[conversation_key] = task

        try:
            return await self._handle_message_impl(conversation_key, text, live_card_callbacks)
        except asyncio.CancelledError:
            logger.info("[%s] Task cancelled by /stop", conversation_key[:8])
            return "⏹ Interrupted."
        finally:
            if task is not None and self._running_tasks.get(conversation_key) is task:
                del self._running_tasks[conversation_key]

    async def _handle_message_impl(
        self, conversation_key: str, text: str, live_card_callbacks: dict[str, Any] | None = None
    ) -> str | None:
        """Internal message handling — separated so handle_message can wrap it
        with task tracking and cancellation support."""
        # Refresh tools each turn (pick up newly written dynamic tools)
        tools = self._refresh_tools()

        harness = await self._get_or_create_harness(conversation_key, tools)

        # Refresh harness tools (pick up newly created dynamic tools)
        await harness.set_tools(tools)

        # Set live card callbacks for real-time Feishu display
        if live_card_callbacks:
            await harness.set_live_card_callbacks(**live_card_callbacks)

        # Get RAG context for this turn
        rag_context = await self._rag.search(text)

        # Build system prompt with fresh data
        system_prompt = self.build_system_prompt(rag_context)
        await harness.set_system_prompt(system_prompt)

        try:
            # Debug
            key = self._config.llm.api_key
            logger.debug("[CODING] api_key=%s model=%s", "***" if key else "MISSING", self._model.id)

            result = await harness.prompt(text)
            if result is None:
                return "No response generated."

            # Debug: log the full result
            logger.debug("[CODING] stop_reason=%s content_blocks=%d",
                         result.stop_reason, len(result.content))
            for i, b in enumerate(result.content):
                t = b.get("type", "?") if isinstance(b, dict) else getattr(b, "type", "?")
                if t == "text":
                    logger.debug("[CODING]   block[%d] text: %s", i, b.get("text", "")[:100])
                elif t == "toolCall":
                    logger.debug("[CODING]   block[%d] toolCall: %s", i, b.get("name", "?"))
                elif t == "thinking":
                    logger.debug("[CODING]   block[%d] thinking: %s", i, b.get("thinking", "")[:100])
            if result.error_message:
                logger.debug("[CODING] error_message: %s", result.error_message)

            text_blocks = [
                c["text"] for c in result.content
                if c.get("type") == "text" and c.get("text")
            ]
            thinking_blocks = [
                c.get("thinking", "") for c in result.content
                if c.get("type") == "thinking"
            ]

            if text_blocks:
                resp = "\n".join(text_blocks)
            elif thinking_blocks:
                # Model returned only thinking, no text — use thinking as response
                logger.debug("[CODING] no text blocks, falling back to thinking (%d blocks)", len(thinking_blocks))
                resp = thinking_blocks[-1][:2000]
            else:
                resp = "(empty response)"

            logger.debug("[CODING] response: %s", resp[:200])
            return resp

        except RuntimeError as e:
            if "busy" in str(e).lower():
                return "I'm still processing your previous request. Please wait."
            raise

    async def new_session(self, conversation_key: str) -> None:
        if conversation_key in self._conversations:
            del self._conversations[conversation_key]
        await self._session_repo.forget_chat(conversation_key)

    async def compact_session(self, conversation_key: str) -> dict | None:
        harness = self._conversations.get(conversation_key)
        if harness:
            return await harness.compact()
        return None

    def abort(self, conversation_key: str | None = None) -> None:
        """Abort the current agent run by cancelling the underlying asyncio task.
        Also sets the abort_event for graceful in-task shutdown as a fallback."""
        keys = [conversation_key] if conversation_key else list(self._conversations.keys())
        for key in keys:
            harness = self._conversations.get(key)
            if harness:
                harness.abort()
            task = self._running_tasks.get(key)
            if task and not task.done():
                task.cancel()

    async def close_conversation(self, conversation_key: str) -> None:
        self._conversations.pop(conversation_key, None)

    async def list_sessions(self) -> list[dict[str, Any]]:
        return await self._session_repo.list_sessions()

    async def initialize_rag(self) -> None:
        await self._rag.initialize()

    # ── Sandbox Escape Authorization ────────────────────────

    async def request_network_auth(self, conversation_key: str, command: str) -> bool:
        """Request user authorization to allow network access for a command."""
        if self._channel is None:
            return False
        return await self._channel.request_network_authorization(conversation_key, command)

    async def request_unsandboxed_auth(self, conversation_key: str, command: str) -> bool:
        """Request user authorization to run a command outside the sandbox entirely."""
        if self._channel is None:
            return False
        return await self._channel.request_unsandboxed_authorization(conversation_key, command)

    # ── Internal ────────────────────────────────────────────

    async def _get_or_create_harness(self, key: str, tools: list[AgentTool] | None = None) -> AgentHarness:
        if tools is None:
            tools = self._tools
        if key not in self._conversations:
            # Load existing session for this chat, or create a new one.
            # Uses chat_id → session_id mapping so context survives restarts.
            session = await self._session_repo.get_or_create_for_chat(key, self._config.agent.cwd)
            harness = AgentHarness(
                session=session,
                model=self._model,
                system_prompt=self.build_system_prompt(),
                tools=tools,
                thinking_level=self._config.agent.thinking_level,  # type: ignore[arg-type]
                compaction_settings=self._compaction_settings,
                get_api_key=lambda _: self._config.llm.api_key or None,
            )

            # ── Wire tool hooks ──────────────────────────

            # Per-turn auth notes so the model sees which tools needed authorization
            _auth_notes: dict[str, list[str]] = {}  # tool_call_id → notes

            async def on_before_tool(ctx: dict, signal=None) -> dict | None:
                """Handle tool authorization: SUSPICIOUS bash, network escape, unsandboxed."""
                tool_call = ctx.get("tool_call", {})
                tc_id = tool_call.get("id", "")
                tool_name = tool_call.get("name", "")
                args = tool_call.get("arguments", {})
                command = args.get("command", "")
                notes: list[str] = []
                _auth_notes[tc_id] = notes

                # Bash tool hooks
                if tool_name == "bash":
                    # 1. Safety check (SUSPICIOUS commands)
                    check = self._bash_guard.check(command)
                    if check == "SUSPICIOUS":
                        notes.append("🔐 Bash authorization")
                        approved = await self._request_bash_auth(key, command)
                        if not approved:
                            notes.append("  → ❌ Denied by user")
                            return {"block": True, "reason": "User denied command execution"}
                        notes.append("  → ✅ Approved by user")

                    # 2. Network escape authorization
                    if args.get("allow_network"):
                        notes.append("🌐 Network access authorization")
                        approved = await self.request_network_auth(key, command)
                        if not approved:
                            notes.append("  → ❌ Denied by user")
                            return {"block": True, "reason": "User denied network access"}
                        notes.append("  → ✅ Approved by user")

                    # 3. Full sandbox escape authorization
                    if args.get("unsandboxed"):
                        notes.append("🚀 Sandbox escape authorization")
                        approved = await self.request_unsandboxed_auth(key, command)
                        if not approved:
                            notes.append("  → ❌ Denied by user")
                            return {"block": True, "reason": "User denied sandbox escape"}
                        notes.append("  → ✅ Approved by user")

                return None

            harness.on("before_tool", on_before_tool)

            async def on_after_tool(ctx: dict, signal=None) -> dict | None:
                """Prepend authorization notes to tool results."""
                tool_call = ctx.get("tool_call", {})
                tc_id = tool_call.get("id", "")
                notes = _auth_notes.pop(tc_id, None)
                if not notes:
                    return None
                result = ctx.get("result")
                if result is None:
                    return None
                prefix = "\n".join(notes) + "\n\n"
                if result.content and result.content[0].get("type") == "text":
                    result.content[0]["text"] = prefix + result.content[0]["text"]
                return None

            harness.on("after_tool", on_after_tool)
            self._conversations[key] = harness

        return self._conversations[key]

    async def _request_bash_auth(self, key: str, command: str) -> bool:
        if self._channel is None:
            return False
        return await self._channel.request_bash_authorization(key, command)
