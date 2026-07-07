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
from connectclaw.memory import MemorySubsystem
from connectclaw.memory.subsystem import MemoryConfig as MemCfg
from connectclaw.provider.types import Model

from .tools.agents import create_agents_tool
from .tools.bash import BashGuard, create_bash_tool
from .tools.hash_edit import create_hash_edit_tool
from .tools.hash_read import create_hash_read_tool
from .tools.image_analyze import create_image_analyze_tool
from .tools import lightpanda
from .tools.read import create_read_tool
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
        self._hash_read_tool = create_hash_read_tool(config.agent.cwd)
        self._hash_edit_tool = create_hash_edit_tool(config.agent.cwd)
        self._bash_guard = BashGuard()
        self._bash_tool = create_bash_tool(config.agent.cwd, self._bash_guard)
        # Size the stateless browser-engine pool (parallel web_search/web_fetch).
        lightpanda.configure_pool(config.web_search.pool_size)
        self._web_search_tool = create_web_search_tool(
            max_chars=config.web_search.max_chars,
            timeout=config.web_search.timeout,
        )
        self._web_fetch_tool = create_web_fetch_tool(
            max_chars=config.web_search.max_chars,
            timeout=config.web_search.timeout,
        )
        self._image_tool = create_image_analyze_tool(
            api_key=config.vision.api_key,
            base_url=config.vision.base_url,
            model_id=config.vision.model_id,
            cwd=config.agent.cwd,
        )

        # Named agents directory (.md agents — the primary "agent makes agent" path)
        self._agents_dir = os.path.expanduser("~/.connectclaw/agents")
        os.makedirs(self._agents_dir, exist_ok=True)

        # Registry of base primitives, keyed by tool name. The main agent's
        # exposed set is a whitelist over this (config.agent.tools); the meta-tool
        # may grant ANY of these to a sub-agent — even ones the main agent lacks.
        self._tool_registry: dict[str, AgentTool] = {
            t.name: t
            for t in [
                self._read_tool, self._write_tool, self._hash_read_tool,
                self._hash_edit_tool, self._bash_tool, self._web_search_tool,
                self._web_fetch_tool, self._image_tool,
            ]
        }

        # The single `agents` meta-tool: list / describe / run / create. Every
        # sub-agent is reached through it, resolved at CALL TIME (so an agent
        # created mid-turn is runnable the same turn). Absorbs the old `task`
        # and `create_agent` tools.
        # Session repository (created before the agents tool so sub-agent
        # transcripts can be persisted through it).
        sessions_dir = os.path.expanduser(config.session.dir)
        self._session_repo = SessionRepo(sessions_dir)

        self._agents_tool = create_agents_tool(
            self._model,
            agents_dir=self._agents_dir,
            base_tools=list(self._tool_registry.values()),
            cwd=config.agent.cwd,
            api_key=config.llm.api_key or None,
            thinking_level=config.agent.thinking_level,  # type: ignore[arg-type]
            session_repo=self._session_repo,
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

        # Memory subsystem (lazy init, all no-ops if disabled)
        self._memory = MemorySubsystem(
            MemCfg(
                enabled=config.memory.enabled,
                db_path=config.memory.db_path,
                extract_after_turn=config.memory.extract_after_turn,
                extract_min_turns=config.memory.extract_min_turns,
                extract_interval_turns=config.memory.extract_interval_turns,
                max_context_tokens=config.memory.max_context_tokens,
                recency_threshold_days=config.memory.recency_threshold_days,
                use_embeddings=config.memory.use_embeddings,
                dream_interval_hours=config.memory.dream_interval_hours,
                decay_halflife_days=config.memory.decay_halflife_days,
                consolidation_enabled=config.memory.consolidation_enabled,
            )
        )

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
    def memory(self) -> MemorySubsystem:
        return self._memory

    @property
    def prompt_builder(self) -> PromptBuilder:
        return self._prompt_builder

    # ── Dynamic Tool Refresh ────────────────────────────────

    def _refresh_tools(self) -> list[AgentTool]:
        """Rebuild the main agent's tools: whitelisted base primitives + the
        single `agents` meta-tool.

        Named agents and dynamic tools are deliberately NOT here — they are
        reached through the meta-tool, which resolves them at CALL TIME. Two
        consequences: (1) the tools array stays byte-stable across turns (it
        changes only when config changes), extending the system-prompt prefix
        cache discipline to the tools array; (2) an agent created mid-turn is
        runnable the same turn, since the meta-tool re-scans on each `run`.
        """
        registry = self._tool_registry
        exposed: list[AgentTool] = []
        for n in self._config.agent.tools:
            tool = registry.get(n)
            if tool is None:
                logger.warning("Unknown tool in [agent].tools whitelist: %r (skipped)", n)
                continue
            exposed.append(tool)
        if not exposed:
            logger.error("[agent].tools resolved to empty; falling back to ['read','bash']")
            exposed = [registry["read"], registry["bash"]]

        tools = exposed + [self._agents_tool]
        self._tools = tools
        return tools

    # ── System Prompt ───────────────────────────────────────

    def build_system_prompt(self) -> str:
        """Build the STABLE system prompt (env + skills only).

        Deliberately takes NO per-turn data. The system prompt must stay
        byte-identical across turns so the provider's prefix cache keeps
        hitting — a single changed token invalidates the whole cache from
        position 0. All volatile context (memory, and RAG once it lands)
        is injected into the user message instead; see _handle_message_impl.
        """
        return self._prompt_builder.build()

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
        # Refresh tools each turn.
        tools = self._refresh_tools()

        harness = await self._get_or_create_harness(conversation_key, tools)

        # Refresh harness tools (pick up newly created .tool.md scripts)
        await harness.set_tools(tools)

        # Set live card callbacks for real-time Feishu display
        if live_card_callbacks:
            await harness.set_live_card_callbacks(**live_card_callbacks)

        # Per-turn dynamic context: RAG (technical docs) + memory (personal).
        # BOTH are injected into the USER MESSAGE, never the system prompt.
        #
        # Why: DeepSeek / OpenAI-compatible providers cache by request *prefix*.
        # Any change to the system prompt invalidates the cache from token 0,
        # so every turn would pay full (uncached) input price. By keeping the
        # system prompt byte-stable and appending the volatile context to the
        # user message — which then persists into history as a fixed prefix for
        # the next turn — the cached prefix keeps growing and keeps hitting.
        rag_context = await self._rag.search(text)
        memory_context = await self._memory.recall(
            text, conversation_key=conversation_key
        )

        # System prompt stays STABLE (no per-turn data) to preserve prefix cache.
        system_prompt = self.build_system_prompt()
        await harness.set_system_prompt(system_prompt)

        try:
            # Debug
            key = self._config.llm.api_key
            logger.debug("[CODING] api_key=%s model=%s", "***" if key else "MISSING", self._model.id)

            # Prepend dynamic context to the user message (memory first — more
            # personal; RAG next — more technical; agent/tool catalog last), all
            # kept OUT of the system prompt to preserve the prefix cache. The
            # catalog is volatile (changes when agents are created), so like
            # memory/RAG it rides in the user message.
            agents_catalog = self._agents_tool.build_catalog()
            context_blocks = [c for c in (memory_context, rag_context, agents_catalog) if c]
            prompt_text = text
            if context_blocks:
                prompt_text = "\n\n".join(context_blocks) + "\n\n" + text

            result = await harness.prompt(prompt_text)
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

            # Extract memories in background (non-blocking, throttled inside)
            if self._memory.enabled:
                try:
                    entries = await harness.session.get_path_to_root()
                    recent_messages = [
                        e.message
                        for e in entries[-20:]
                        if hasattr(e, "type") and e.type == "message"
                    ]
                    if recent_messages:
                        asyncio.create_task(
                            self._memory.learn(
                                recent_messages,
                                self._model,
                                api_key=self._config.llm.api_key or None,
                                conversation_key=conversation_key,
                                session_id=harness.session.session_id,
                            )
                        )
                except Exception as e:
                    logger.debug("Memory extraction trigger failed: %s", e)

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
