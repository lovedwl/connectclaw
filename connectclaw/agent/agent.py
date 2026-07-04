"""
Agent class — stateful wrapper around the agent loop.

Owns transcript, emits lifecycle events, executes tools.
Mirrors pi-mono's Agent class with Python async patterns.
"""

import asyncio
import time
from typing import Any, Callable

from connectclaw.provider.stream import stream_simple
from connectclaw.provider.types import Model, UserMessage

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolResult,
    QueueMode,
    ThinkingLevel,
    ToolExecutionMode,
)


def _default_convert_to_llm(messages: list[AgentMessage]) -> list:
    """Keep only LLM-compatible messages."""
    return [m for m in messages if m.role in ("user", "assistant", "toolResult")]


class Agent:
    """Stateful wrapper around the low-level agent loop."""

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: Model | None = None,
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool] | None = None,
        messages: list[AgentMessage] | None = None,
        convert_to_llm: Callable | None = None,
        transform_context: Callable | None = None,
        stream_fn: Callable | None = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        tool_execution: ToolExecutionMode = "parallel",
        session_id: str | None = None,
        get_api_key: Callable | None = None,
        max_retry_delay_ms: int = 60_000,
    ):
        self._state = AgentState(
            system_prompt=system_prompt,
            model=model,
            thinking_level=thinking_level,
            tools=tools or [],
            messages=messages or [],
        )
        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._steering_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._follow_up_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self._tool_execution = tool_execution
        self._session_id = session_id
        self._get_api_key = get_api_key
        self._max_retry_delay_ms = max_retry_delay_ms
        self.convert_to_llm = convert_to_llm or _default_convert_to_llm
        self.transform_context = transform_context
        self.stream_fn = stream_fn or stream_simple
        self._live_card: dict[str, Any] = {}

        self._active_run: dict[str, Any] | None = None

    # ── Properties ──────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._session_id = value

    @property
    def tool_execution(self) -> ToolExecutionMode:
        return self._tool_execution  # type: ignore[return-type]

    @tool_execution.setter
    def tool_execution(self, value: ToolExecutionMode) -> None:
        self._tool_execution = value

    # ── Public API ──────────────────────────────────────────

    async def prompt(self, input_: AgentMessage | list[AgentMessage] | str) -> None:
        """Start a new conversation turn. Raises if already running."""
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing a prompt")

        if isinstance(input_, str):
            messages: list[AgentMessage] = [
                UserMessage(content=input_, timestamp=time.time() * 1000)
            ]  # type: ignore[list-item]
        elif isinstance(input_, list):
            messages = input_
        else:
            messages = [input_]

        await self._run_prompt(messages)

    async def continue_(self) -> None:
        """Continue from current context. Last message must be user/toolResult."""
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing")

        messages = self._state.messages
        if not messages:
            raise ValueError("No messages to continue from")

        last = messages[-1]
        if last.role == "assistant":
            # Check for queued steering/follow-up
            queued_steering = await self._drain_steering()
            if queued_steering:
                await self._run_prompt(queued_steering)
                return

            queued_follow_up = await self._drain_follow_up()
            if queued_follow_up:
                await self._run_prompt(queued_follow_up)
                return

            raise ValueError("Cannot continue from assistant message without queued messages")

        await self._continue_current()

    def steer(self, message: AgentMessage) -> None:
        """Queue a steering message for the current run."""
        self._steering_queue.put_nowait(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a follow-up message for after the agent stops."""
        self._follow_up_queue.put_nowait(message)

    def abort(self) -> None:
        """Signal abort to the current run."""
        if self._active_run:
            self._active_run["abort_event"].set()

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to agent events. Returns unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    async def reset(self) -> None:
        await self.wait_for_idle()
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        # Clear queues
        while not self._steering_queue.empty():
            self._steering_queue.get_nowait()
        while not self._follow_up_queue.empty():
            self._follow_up_queue.get_nowait()

    async def wait_for_idle(self) -> None:
        if self._active_run:
            await self._active_run["finish"]

    # ── State Mutators ──────────────────────────────────────

    def set_system_prompt(self, value: str) -> None:
        self._state.system_prompt = value

    def set_model(self, model: Model) -> None:
        self._state.model = model

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._state.thinking_level = level

    def set_live_card_callbacks(self, **callbacks: Any) -> None:
        """Set live card callbacks for real-time Feishu display."""
        self._live_card = callbacks

    def set_tools(self, tools: list[AgentTool]) -> None:
        self._state.tools = tools

    def set_steering_mode(self, mode: QueueMode) -> None:
        self._steering_mode = mode

    def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_mode = mode

    def clear_queues(self) -> None:
        while not self._steering_queue.empty():
            self._steering_queue.get_nowait()
        while not self._follow_up_queue.empty():
            self._follow_up_queue.get_nowait()

    def has_queued_messages(self) -> bool:
        return not self._steering_queue.empty() or not self._follow_up_queue.empty()

    # ── Internal ────────────────────────────────────────────

    async def _drain_steering(self) -> list[AgentMessage]:
        if self._steering_mode == "one-at-a-time":
            try:
                msg = self._steering_queue.get_nowait()
                return [msg]
            except asyncio.QueueEmpty:
                return []

        # Drain all
        messages = []
        while True:
            try:
                messages.append(self._steering_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    async def _drain_follow_up(self) -> list[AgentMessage]:
        if self._follow_up_mode == "one-at-a-time":
            try:
                msg = self._follow_up_queue.get_nowait()
                return [msg]
            except asyncio.QueueEmpty:
                return []

        messages = []
        while True:
            try:
                messages.append(self._follow_up_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    async def _run_prompt(self, messages: list[AgentMessage]) -> None:
        abort_event = asyncio.Event()
        finish: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active_run = {
            "abort_event": abort_event,
            "finish": finish,
        }

        self._state.is_streaming = True
        self._state.error_message = None

        try:
            await run_agent_loop(
                messages,
                self._build_context(),
                self._build_loop_config(abort_event),
                self._emit_event,
                abort_event,
            )
        except Exception as e:
            self._state.error_message = str(e)
            self._emit_sync({"type": "agent_end", "messages": []})
        finally:
            self._state.is_streaming = False
            self._state.streaming_message = None
            self._state.pending_tool_calls = set()
            finish.set_result(None)
            self._active_run = None

    async def _continue_current(self) -> None:
        abort_event = asyncio.Event()
        finish: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active_run = {
            "abort_event": abort_event,
            "finish": finish,
        }

        self._state.is_streaming = True
        self._state.error_message = None

        try:
            await run_agent_loop_continue(
                self._build_context(),
                self._build_loop_config(abort_event),
                self._emit_event,
                abort_event,
            )
        except Exception as e:
            self._state.error_message = str(e)
            self._emit_sync({"type": "agent_end", "messages": []})
        finally:
            self._state.is_streaming = False
            self._state.streaming_message = None
            self._state.pending_tool_calls = set()
            finish.set_result(None)
            self._active_run = None

    def _build_context(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=self._state.tools,
        )

    def _build_loop_config(self, abort_event: asyncio.Event) -> AgentLoopConfig:
        reasoning = None
        if self._state.thinking_level != "off":
            reasoning = self._state.thinking_level

        lc = self._live_card
        return AgentLoopConfig(
            model=self._state.model,  # type: ignore[arg-type]
            reasoning=reasoning,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self._get_api_key,
            get_steering_messages=self._drain_steering,
            get_follow_up_messages=self._drain_follow_up,
            tool_execution=self._tool_execution,
            session_id=self._session_id,
            max_retry_delay_ms=self._max_retry_delay_ms,
            on_thinking_delta=lc.get("on_thinking_delta"),
            on_thinking_done=lc.get("on_thinking_done"),
            on_tool_call=lc.get("on_tool_call"),
            on_tool_result=lc.get("on_tool_result"),
            on_text_delta=lc.get("on_text_delta"),
            on_text_done=lc.get("on_text_done"),
        )

    async def _emit_event(self, event: AgentEvent) -> None:
        self._reduce_event(event)
        for listener in self._listeners:
            result = listener(event)
            if asyncio.iscoroutine(result):
                await result

    def _emit_sync(self, event: AgentEvent) -> None:
        self._reduce_event(event)
        for listener in self._listeners:
            listener(event)

    def _reduce_event(self, event: AgentEvent) -> None:
        """Update agent state based on event."""
        etype = event.get("type", "")
        match etype:
            case "message_start":
                self._state.streaming_message = event.get("message")
            case "message_update":
                self._state.streaming_message = event.get("message")
            case "message_end":
                msg = event.get("message")
                if msg:
                    self._state.messages.append(msg)
                self._state.streaming_message = None
            case "tool_execution_start":
                tid = event.get("tool_call_id", "")
                self._state.pending_tool_calls.add(tid)
            case "tool_execution_end":
                tid = event.get("tool_call_id", "")
                self._state.pending_tool_calls.discard(tid)
            case "turn_end":
                msg = event.get("message", {})
                if isinstance(msg, dict) and msg.get("error_message"):
                    self._state.error_message = msg["error_message"]
            case "agent_end":
                self._state.is_streaming = False
                self._state.streaming_message = None
