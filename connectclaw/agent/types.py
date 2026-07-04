"""Agent layer types for ConnectClaw."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from connectclaw.provider.types import Message, Model


# ── Thinking Level ────────────────────────────────────────────

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

# ── Queue Mode ─────────────────────────────────────────────────

QueueMode = Literal["all", "one-at-a-time"]

# ── Tool Execution Mode ────────────────────────────────────────

ToolExecutionMode = Literal["sequential", "parallel"]


# ── Agent Tool ─────────────────────────────────────────────────


@dataclass
class AgentToolResult:
    """Result from executing an AgentTool."""
    content: list[dict[str, Any]] = field(default_factory=list)
    details: Any = None
    terminate: bool = False  # hint to stop after this tool batch


AgentToolUpdateCallback = Callable[["AgentToolResult"], None]


class AgentTool:
    """Tool definition used by the agent runtime."""

    name: str = ""
    label: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    execution_mode: ToolExecutionMode = "parallel"

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: "asyncio.Event | None" = None,
        on_update: AgentToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        raise NotImplementedError


# ── Agent Messages ─────────────────────────────────────────────


@dataclass
class BashExecutionMessage:
    role: Literal["bashExecution"] = "bashExecution"
    command: str = ""
    output: str = ""
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    timestamp: float = 0.0


@dataclass
class CompactionSummaryMessage:
    role: Literal["compactionSummary"] = "compactionSummary"
    summary: str = ""
    tokens_before: int = 0
    timestamp: float = 0.0


@dataclass
class BranchSummaryMessage:
    role: Literal["branchSummary"] = "branchSummary"
    summary: str = ""
    from_id: str = ""
    timestamp: float = 0.0


AgentMessage = (
    Message | BashExecutionMessage | CompactionSummaryMessage | BranchSummaryMessage
)


# ── Agent Events ───────────────────────────────────────────────

AgentEvent = dict[str, Any]
# {
#   "type": "agent_start" | "agent_end" | "turn_start" | "turn_end"
#        | "message_start" | "message_update" | "message_end"
#        | "tool_execution_start" | "tool_execution_update" | "tool_execution_end"
#   ... event-specific fields
# }


# ── Agent State ────────────────────────────────────────────────


@dataclass
class AgentState:
    system_prompt: str = ""
    model: Model | None = None
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    streaming_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: str | None = None


# ── Agent Context ──────────────────────────────────────────────


@dataclass
class AgentContext:
    system_prompt: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[AgentTool] | None = None


# ── Agent Loop Config ──────────────────────────────────────────


@dataclass
class AgentLoopConfig:
    model: Model
    reasoning: str | None = None  # thinking level for the model
    convert_to_llm: Callable[[list[AgentMessage]], list[Message]] | None = None
    transform_context: (
        Callable[[list[AgentMessage], "asyncio.Event | None"], Awaitable[list[AgentMessage]]]
        | None
    ) = None
    get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = None
    api_key: str | None = None
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: (
        Callable[[dict[str, Any], "asyncio.Event | None"], Awaitable[dict[str, Any] | None]]
        | None
    ) = None
    after_tool_call: (
        Callable[[dict[str, Any], "asyncio.Event | None"], Awaitable[dict[str, Any] | None]]
        | None
    ) = None
    session_id: str | None = None
    transport: str = "sse"
    max_retry_delay_ms: int = 60_000
    max_retries: int = 3
    # Live card callbacks (Phase 1 thinking card + Phase 2 text streaming)
    on_thinking_delta: Callable[[str], Awaitable[None]] | None = None
    on_thinking_done: Callable[[float], Awaitable[None]] | None = None
    on_tool_call: Callable[[str, dict], Awaitable[None]] | None = None
    on_tool_result: Callable[[str, bool, str], Awaitable[None]] | None = None
    # (tool_name, is_error, result_text)
    on_text_delta: Callable[[str], Awaitable[None]] | None = None
    on_text_done: Callable[[], Awaitable[None]] | None = None
