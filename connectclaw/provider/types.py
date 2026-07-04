"""Core types for the ConnectClaw provider layer."""

from dataclasses import dataclass, field
from typing import Any, Literal

# ── Message Types ──────────────────────────────────────────────


@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: str | list[dict[str, Any]] = ""
    timestamp: float = 0.0


@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    stop_reason: Literal["stop", "length", "toolUse", "error", "aborted"] = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    error_message: str | None = None
    timestamp: float = 0.0


@dataclass
class ToolResultMessage:
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = ""
    tool_name: str = ""
    content: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    timestamp: float = 0.0


Message = UserMessage | AssistantMessage | ToolResultMessage


def normalize_message(raw: dict[str, Any] | Message) -> Message:
    """Convert a dict message to the appropriate dataclass, or pass through if already one.

    Call this at entry points (session load, agent.prompt) so the rest of
    the codebase only deals with dataclass instances and never needs hasattr/getattr.
    """
    if not isinstance(raw, dict):
        return raw  # already a dataclass

    role = raw.get("role", "")
    if role == "assistant":
        return AssistantMessage(
            content=raw.get("content", []),
            model=raw.get("model", ""),
            stop_reason=raw.get("stop_reason", "stop"),
            usage=raw.get("usage", {}),
            error_message=raw.get("error_message"),
            timestamp=raw.get("timestamp", 0.0),
        )
    if role == "toolResult":
        return ToolResultMessage(
            tool_call_id=raw.get("tool_call_id", ""),
            tool_name=raw.get("tool_name", ""),
            content=raw.get("content", []),
            is_error=raw.get("is_error", False),
            timestamp=raw.get("timestamp", 0.0),
        )
    # user or unknown → UserMessage
    content = raw.get("content", "")
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ) or content
    return UserMessage(
        content=content,
        timestamp=raw.get("timestamp", 0.0),
    )


# ── Model & Tool ───────────────────────────────────────────────


@dataclass
class Model:
    id: str
    name: str = ""
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    api: str = "openai-compatible"
    reasoning: bool = True
    context_window: int = 65536
    max_tokens: int = 8192


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


# ── Context ────────────────────────────────────────────────────


@dataclass
class Context:
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)
    tools: list[ToolDef] | None = None


# ── Stream Events ──────────────────────────────────────────────

StreamEventType = Literal[
    "start",
    "text_delta",
    "thinking_delta",
    "toolcall_delta",
    "done",
    "error",
]


@dataclass
class StreamEvent:
    type: StreamEventType
    delta: str | None = None
    content_index: int = 0
    partial: AssistantMessage | None = None
    message: AssistantMessage | None = None
    error_message: str | None = None
