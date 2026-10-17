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
from dataclasses import asdict
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
    should_compact,
    prepare_compaction,
    compact as entry_based_compact,
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
    before_compact: list[
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
        # H5：强引用持有落盘任务。CPython 事件循环只持弱引用，
        # fire-and-forget 的 create_task 可能在执行中途被 GC（消息静默丢失）。
        self._persist_tasks: set[asyncio.Task] = set()

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
        event: Literal["before_agent_start", "before_llm_call", "before_tool", "after_tool", "context", "before_compact"],
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

    def _active_abort_event(self) -> asyncio.Event | None:
        """M10：当前运行中的 abort 事件（/stop 信号源）；不在 turn 中则 None。"""
        agent = self._agent
        if agent is None:
            return None
        active = getattr(agent, "_active_run", None)
        if isinstance(active, dict):
            return active.get("abort_event")
        return None

    def _on_persist_done(self, task: asyncio.Task) -> None:
        """H5：落盘任务完成后从强引用集合移除；异常必须 retrieve（防静默丢失）。"""
        self._persist_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Session persist failed", exc_info=exc)

    async def prompt(self, text: str, images: list[dict] | None = None) -> AssistantMessage | None:
        """Execute a prompt turn. Raises if busy.

        ``images`` are image_ref blocks attached to this turn's user message
        (resolved to image_url only for the current turn by the provider).
        """
        if self._phase != "idle":
            raise RuntimeError("AgentHarness is busy")

        self._phase = "turn"
        try:
            return await self._execute_turn(text, images=images)
        finally:
            self._phase = "idle"

    @property
    def turn_aborted(self) -> bool:
        """最近一次 prompt 是否被 /stop 中断（见 agent_loop 的 aborted 收尾）。"""
        return getattr(self, "_turn_aborted", False)

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

    def abort(self) -> None:
        """Signal the current agent run to stop."""
        if self._agent:
            self._agent.abort()

    async def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        if self._agent:
            self._agent.set_system_prompt(prompt)

    async def append_context(self, ops: list[dict[str, Any]], note: str = "") -> str:
        """落盘一条结构化注入条目（本轮增量；渲染只发生在组装上下文那一刻）。"""
        return await self._session.append_context(ops, note=note)

    async def context_state(self) -> Any:
        """当前上下文状态（结构化对象）——**从会话折叠出来**，不依赖任何内存账本。

        这是"哪些已经注入过"的唯一事实来源：重启后重新折叠即可，所以不会出现
        "进程一重启就把 persona/清单/召回整批重发一遍"。

        折叠顺序跟消息层一致：遇到压缩条目就按 ``first_kept_entry_id`` 丢掉被压缩掉的
        批次，并把压缩快照作为**基**放在保留区之前（文件里压缩条目排在保留区之后，
        顺序照搬会算错）。
        """
        from connectclaw.injection import fold_ops

        entries = await self._session.get_path_to_root()
        collected: list[tuple[str, list[dict[str, Any]]]] = []

        for entry in entries:
            etype = getattr(entry, "type", "")
            eid = getattr(entry, "id", "")

            if etype == "context":
                ops = list(getattr(entry, "ops", []) or [])
                if ops:
                    collected.append((eid, ops))

            elif etype == "compaction":
                snapshot = getattr(entry, "context_state", None)
                batch = [snapshot] if snapshot else []
                kept_from = getattr(entry, "first_kept_entry_id", "")
                idx = next(
                    (i for i, (cid, _) in enumerate(collected) if cid == kept_from), None
                )
                kept = collected[idx:] if idx is not None else []
                collected = ([(eid, batch)] if batch else []) + kept

        return fold_ops([ops for _, ops in collected])

    async def compact(self, custom_instructions: str | None = None) -> dict[str, Any] | None:
        """Trigger manual compaction using entry-based pipeline."""
        if not self._agent:
            return None

        entries = await self._session.get_path_to_root()
        if not entries:
            return None

        api_key = None
        if self._get_api_key:
            result = self._get_api_key(self._model.provider)
            if asyncio.iscoroutine(result):
                api_key = await result
            else:
                api_key = result

        entries_dicts = [asdict(e) for e in entries]
        prep = prepare_compaction(
            entries_dicts, self._compaction_settings, context_window=self._model.context_window
        )
        if not prep:
            return {"summary": None, "tokens_before": 0}

        result = await entry_based_compact(
            prep, self._model,
            api_key=api_key,
            custom_instructions=custom_instructions,
            thinking_level=self._thinking_level,
            settings=self._compaction_settings,
            signal=self._active_abort_event(),
        )

        # M10 后续：abort 中断的摘要可能是半截/降级原文——不能落盘。
        abort_event = self._active_abort_event()
        aborted = abort_event is not None and abort_event.is_set()
        if result and not aborted:
            await self._session.append_compaction(
                result.summary, result.first_kept_entry_id, result.tokens_before,
                merged_context=result.merged_context,
                context_state=result.context_state,
            )

        return {
            "summary": result.summary if result else None,
            "tokens_before": result.tokens_before if result else 0,
        }

    # ── Internal ────────────────────────────────────────────

    async def _execute_turn(
        self, text: str, images: list[dict] | None = None
    ) -> AssistantMessage | None:
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

        # Resolve API key for compaction
        compaction_api_key: str | None = None
        if self._get_api_key:
            key_result = self._get_api_key(self._model.provider)
            if asyncio.iscoroutine(key_result):
                compaction_api_key = await key_result
            else:
                compaction_api_key = key_result

        # Check if compaction is needed
        tokens = calculate_context_tokens(messages)
        if should_compact(tokens, self._model.context_window, self._compaction_settings):
            # Fire before_compact hook so memory extraction can happen
            # while raw conversation detail is still available.
            for hook in self._hooks.before_compact:
                result = hook({
                    "tokens": tokens,
                    "messages": messages,
                })
                if asyncio.iscoroutine(result):
                    await result

            # Use entry-based pipeline so first_kept_entry_id is properly set
            entries_dicts = [asdict(e) for e in entries]
            prep = prepare_compaction(
                entries_dicts,
                self._compaction_settings,
                context_window=self._model.context_window,
            )
            if prep:
                result = await entry_based_compact(
                    prep, self._model,
                    api_key=compaction_api_key,
                    thinking_level=self._thinking_level,
                    settings=self._compaction_settings,
                    signal=self._active_abort_event(),
                )
                # M10 后续：abort 中断的半截/降级摘要不落盘。
                abort_event = self._active_abort_event()
                if result and not (abort_event is not None and abort_event.is_set()):
                    await self._session.append_compaction(
                        result.summary, result.first_kept_entry_id, result.tokens_before,
                        merged_context=result.merged_context,
                        context_state=result.context_state,
                    )
                    if not prep.fits_budget:
                        logger.warning(
                            "compaction under-delivered: freed~%d, post-compact est~%d "
                            "(budget %d) — recent window at floor; summary may still be large",
                            prep.liberated_tokens,
                            prep.post_compact_estimate,
                            self._model.context_window - self._compaction_settings.reserve_tokens,
                        )
                    # Rebuild context after compaction
                    entries = await self._session.get_path_to_root()
                    ctx = build_session_context(entries)
                    messages = list(ctx.messages)

        # Wrap harness hooks into callables for the agent loop
        _before_tool_hooks = list(self._hooks.before_tool)
        _after_tool_hooks = list(self._hooks.after_tool)

        async def _before_tool_wrapper(ctx: dict, signal=None) -> dict | None:
            for hook in _before_tool_hooks:
                result = hook(ctx, signal)
                if asyncio.iscoroutine(result):
                    result = await result
                if result and result.get("block"):
                    return result
            return None

        async def _after_tool_wrapper(ctx: dict, signal=None) -> dict | None:
            for hook in _after_tool_hooks:
                result = hook(ctx, signal)
                if asyncio.iscoroutine(result):
                    result = await result
                if result:
                    return result
            return None

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
                before_tool_call=_before_tool_wrapper if _before_tool_hooks else None,
                after_tool_call=_after_tool_wrapper if _after_tool_hooks else None,
            )
            self._agent.set_live_card_callbacks(**self._live_card_callbacks)
        else:
            # Update existing agent state for the new turn
            self._agent.set_system_prompt(system_prompt)
            self._agent.set_model(self._model)
            self._agent.set_thinking_level(self._thinking_level)
            self._agent.set_tools(self._tools)
            # Update hooks (may have changed since last turn)
            self._agent._before_tool_call = _before_tool_wrapper if _before_tool_hooks else None
            self._agent._after_tool_call = _after_tool_wrapper if _after_tool_hooks else None
            # Reset messages to session context
            self._agent.state.messages = list(messages)

        agent = self._agent

        # Subscribe to agent events (only once, avoid duplicate listeners)
        if not hasattr(self, '_harness_subscribed'):
            def on_agent_event(event: dict) -> None:
                etype = event.get("type", "")

                # /stop 中断标记：turn_end 携带 stop_reason="aborted" 的收尾消息
                if etype == "turn_end":
                    msg = event.get("message")
                    sr = getattr(msg, "stop_reason", None)
                    if sr is None and isinstance(msg, dict):
                        sr = msg.get("stop_reason")
                    self._turn_aborted = (sr == "aborted")

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
                            task = asyncio.get_running_loop().create_task(
                                self._session.append_message(_serialize_message(msg))
                            )
                        except RuntimeError:
                            pass  # No running loop, skip
                        else:
                            self._persist_tasks.add(task)
                            task.add_done_callback(self._on_persist_done)

            agent.subscribe(on_agent_event)
            self._harness_subscribed = True

        # Run the prompt
        if images:
            prompt_msg = UserMessage(
                content=[{"type": "text", "text": text}, *images],
                timestamp=time.time() * 1000,
            )
        else:
            prompt_msg = text
        self._turn_aborted = False
        await agent.prompt(prompt_msg)

        # H5：等本轮 message_end 落盘任务完成再返回，保证 turn 边界上会话文件
        # 已完整（下一轮折叠/压缩读文件时不会缺尾部消息，也不会与下轮写入乱序）。
        if self._persist_tasks:
            await asyncio.gather(*list(self._persist_tasks), return_exceptions=True)

        # Return last assistant message
        # M8：被 abort 的轮次没有本轮 assistant 消息（abort 不 emit
        # message_end）——从尾部找到的只会是**上一条**旧消息，返回它是
        # 契约错误。abort 时返回 None。
        if self._turn_aborted:
            logger.info("Turn aborted — prompt() returns None (no fresh assistant message)")
            return None

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
                duration_ms=getattr(m, 'duration_ms', 0.0),
                ttft_ms=getattr(m, 'ttft_ms', 0.0),
                retry_count=getattr(m, 'retry_count', 0),
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