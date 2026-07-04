"""
AgentHarness — higher-level orchestrator wrapping Agent.

Manages session lifecycle, compaction, skills, prompt templates,
system prompt construction, and pending writes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from connectclaw.agent.agent import Agent
from connectclaw.agent.types import (
    AgentMessage,
    AgentTool,
    AgentToolResult,
    ThinkingLevel,
)
from connectclaw.logging import get_logger
from connectclaw.provider.types import (
    AssistantMessage,
    Message,
    Model,
    UserMessage,
)

logger = get_logger(__name__)

from .compaction import (
    CompactionSettings,
    calculate_context_tokens,
    compact_conversation,
    estimate_tokens,
    should_compact,
)
from .messages import convert_to_llm
from .session import (
    JsonlSessionStorage,
    SessionContext,
    SessionRepo,
    build_session_context,
)


# ── Harness Phase ──────────────────────────────────────────────

HarnessPhase = Literal["idle", "turn"]


# ── Hooks ──────────────────────────────────────────────────────

@dataclass
class AgentHarnessHooks:
    before_agent_start: list[
        Callable[[dict[str, Any]], dict[str, Any] | None]
    ] = field(default_factory=list)
    before_llm_call: list[
        Callable[[dict[str, Any]], dict[str, Any] | None]
    ] = field(default_factory=list)
    before_tool: list[
        Callable[[dict[str, Any]], dict[str, Any] | None]
    ] = field(default_factory=list)
    after_tool: list[
        Callable[[dict[str, Any]], dict[str, Any] | None]
    ] = field(default_factory=list)
    context: list[
        Callable[[dict[str, Any]], None]
    ] = field(default_factory=list)


# ── AgentHarness ───────────────────────────────────────────────


@dataclass
class HarnessState:
    phase: HarnessPhase = "idle"
    session: JsonlSessionStorage | None = None
    system_prompt: str = ""
    model: Model | None = None
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)


class AgentHarness:
    """High-level orchestrator wrapping Agent."""

    def __init__(
        self,
        *,
        session: JsonlSessionStorage,
        model: Model,
        system_prompt: str | Callable[[dict], str] = "",
        tools: list[AgentTool] | None = None,
        thinking_level: ThinkingLevel = "off",
        compaction_settings: CompactionSettings | None = None,
        get_api_key: Callable | None = None,
    ):
        self._session = session
        self._model = model
        self._thinking_level = thinking_level
        self._compaction_settings = compaction_settings or CompactionSettings()
        self._get_api_key = get_api_key
        self._hooks = AgentHarnessHooks()
        self._live_card_callbacks: dict[str, Any] = {}

        # Resolve system prompt
        if callable(system_prompt):
            self._system_prompt_fn = system_prompt
            self._system_prompt = ""
        else:
            self._system_prompt_fn = None
            self._system_prompt = system_prompt

        self._tools = tools or []
        self._agent: Agent | None = None
        self._phase: HarnessPhase = "idle"

        # Subscribers for harness events
        self._listeners: list[Callable] = []

    # ── Properties ──────────────────────────────────────────

    @property
    def phase(self) -> HarnessPhase:
        return self._phase

    @property
    def model(self) -> Model:
        return self._model

    @property
    def session(self) -> JsonlSessionStorage:
        return self._session

    # ── Hooks ───────────────────────────────────────────────

    def on(
        self,
        event: Literal["before_agent_start", "before_llm_call", "before_tool", "after_tool", "context"],
        handler: Callable,
    ) -> Callable[[], None]:
        hook_list = getattr(self._hooks, event, None)
        if hook_list is not None:
            hook_list.append(handler)
            return lambda: hook_list.remove(handler)
        return lambda: None

    def subscribe(self, listener: Callable) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    # ── Prompt API ──────────────────────────────────────────

    async def prompt(self, text: str) -> AssistantMessage | None:
        """Execute a prompt turn. Raises if busy."""
        if self._phase != "idle":
            raise RuntimeError("AgentHarness is busy")

        self._phase = "turn"
        try:
            return await self._execute_turn(text)
        finally:
            self._phase = "idle"

    async def steer(self, text: str) -> None:
        """Queue a steering message."""
        msg = UserMessage(content=text, timestamp=time.time() * 1000)
        if self._agent:
            self._agent.steer(msg)  # type: ignore[arg-type]

    async def follow_up(self, text: str) -> None:
        """Queue a follow-up message."""
        msg = UserMessage(content=text, timestamp=time.time() * 1000)
        if self._agent:
            self._agent.follow_up(msg)  # type: ignore[arg-type]

    # ── State Management ────────────────────────────────────

    async def set_model(self, model: Model) -> None:
        self._model = model
        if self._agent:
            self._agent.set_model(model)

    async def set_live_card_callbacks(self, **callbacks: Any) -> None:
        self._live_card_callbacks = callbacks
        if self._agent:
            self._agent.set_live_card_callbacks(**callbacks)

    async def set_tools(self, tools: list[AgentTool]) -> None:
        self._tools = tools
        if self._agent:
            self._agent.set_tools(tools)

    async def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        if self._agent:
            self._agent.set_system_prompt(prompt)

    async def compact(self, custom_instructions: str | None = None) -> dict[str, Any] | None:
        """Trigger manual compaction."""
        if not self._agent:
            return None

        messages = list(self._agent.state.messages)
        if not messages:
            return None

        result = await compact_conversation(
            messages,
            self._model,
            self._compaction_settings,
            self._model.context_window,
            thinking_level=self._thinking_level,
        )

        if result:
            await self._session.append_compaction(
                result.summary, result.first_kept_entry_id, result.tokens_before
            )

        return {
            "summary": result.summary if result else None,
            "tokens_before": result.tokens_before if result else 0,
        }

    # ── Internal ────────────────────────────────────────────

    async def _execute_turn(self, text: str) -> AssistantMessage | None:
        """Actually execute the agent loop for one prompt."""
        # Build system prompt
        system_prompt = self._system_prompt
        if self._system_prompt_fn:
            system_prompt = self._system_prompt_fn({})

        # Rebuild session context (may have compaction entries)
        entries = await self._session.get_path_to_root()
        ctx = build_session_context(entries)
        messages = list(ctx.messages)

        # Apply hooks: before_agent_start
        for hook in self._hooks.before_agent_start:
            result = hook({"text": text, "messages": messages})
            if result and "messages" in result:
                messages = result["messages"]

        # Check if compaction is needed
        tokens = calculate_context_tokens(messages)
        if should_compact(tokens, self._model.context_window, self._compaction_settings):
            result = await compact_conversation(
                messages,
                self._model,
                self._compaction_settings,
                self._model.context_window,
                thinking_level=self._thinking_level,
            )
            if result:
                await self._session.append_compaction(
                    result.summary, result.first_kept_entry_id, result.tokens_before
                )
                # Rebuild context after compaction
                entries = await self._session.get_path_to_root()
                ctx = build_session_context(entries)
                messages = list(ctx.messages)

        # Create or reuse the agent
        if self._agent is None:
            self._agent = Agent(
                system_prompt=system_prompt,
                model=self._model,
                thinking_level=self._thinking_level,
                tools=self._tools,
                messages=messages,
                convert_to_llm=convert_to_llm,
                get_api_key=self._get_api_key,
            )
            self._agent.set_live_card_callbacks(**self._live_card_callbacks)
        else:
            # Update existing agent state for the new turn
            self._agent.set_system_prompt(system_prompt)
            self._agent.set_model(self._model)
            self._agent.set_thinking_level(self._thinking_level)
            self._agent.set_tools(self._tools)
            # Reset messages to session context
            self._agent.state.messages = list(messages)

        agent = self._agent

        # Subscribe to agent events (only once, avoid duplicate listeners)
        if not hasattr(self, '_harness_subscribed'):
            def on_agent_event(event: dict) -> None:
                etype = event.get("type", "")

                # Emit to harness listeners
                for listener in self._listeners:
                    try:
                        listener(event)
                    except Exception:
                        logger.warning("Listener error in %s event", etype, exc_info=True)

                # Auto-save messages to session
                if etype == "message_end":
                    msg = event.get("message")
                    if msg:
                        try:
                            asyncio.get_running_loop().create_task(
                                self._session.append_message(_serialize_message(msg))
                            )
                        except RuntimeError:
                            pass  # No running loop, skip

            agent.subscribe(on_agent_event)
            self._harness_subscribed = True

        # Run the prompt
        await agent.prompt(text)

        # Return last assistant message — reconstruct from dict or dataclass
        for m in reversed(agent.state.messages):
            if m.role != "assistant":
                continue

            return AssistantMessage(
                content=m.content,
                model=self._model.id,
                stop_reason=m.stop_reason if hasattr(m, 'stop_reason') else "stop",
                usage=m.usage if hasattr(m, 'usage') else {},
                error_message=m.error_message if hasattr(m, 'error_message') else None,
                timestamp=time.time() * 1000,
            )

        logger.warning("No assistant message found in agent state after prompt")
        return None


def _serialize_message(msg) -> dict[str, Any]:
    """Serialize a message to a JSON-safe dict."""
    if hasattr(msg, "__dict__"):
        d = dict(msg.__dict__)
        # Handle nested objects
        for key, val in list(d.items()):
            if hasattr(val, "__dict__"):
                d[key] = val.__dict__
        return d
    if isinstance(msg, dict):
        return dict(msg)
    return {"content": str(msg)}
