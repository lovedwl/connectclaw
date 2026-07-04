"""
Context window compaction — pi-mono parity.

Pipeline:
  estimate tokens (provider usage anchor) → should compact?
  → prepare (find cut point, gather entries, check split-turn)
  → summarize via LLM (full or incremental)
  → persist compaction entry (tree structure)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from connectclaw.provider.types import Context, Model
from connectclaw.provider.stream import stream_simple

from ..types import AgentMessage, ThinkingLevel


# ── Settings ───────────────────────────────────────────────────


@dataclass
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000


# ── Token Estimation ───────────────────────────────────────────


def estimate_tokens(message: Any) -> int:
    """Estimate token count using chars/4 heuristic, per-message-type."""
    msg = _unwrap(message)
    role = msg.get("role", "")

    if role == "user":
        content = msg.get("content", "")
        if isinstance(content, str):
            return max(1, len(content) // 4)
        if isinstance(content, list):
            chars = sum(len(str(b.get("text", ""))) for b in content)
        else:
            chars = len(str(content))
        return max(1, chars // 4)

    if role == "assistant":
        content = msg.get("content", [])
        if isinstance(content, str):
            return max(1, len(content) // 4)
        chars = 0
        for block in content:
            if not isinstance(block, dict):
                chars += len(str(block))
                continue
            t = block.get("type", "")
            if t == "text":
                chars += len(str(block.get("text", "")))
            elif t == "thinking":
                chars += len(str(block.get("thinking", "")))
            elif t == "toolCall":
                chars += len(str(block.get("name", ""))) + len(str(block.get("arguments", {})))
        return max(1, chars // 4)

    if role == "toolResult":
        chars = sum(len(str(b.get("text", ""))) for b in msg.get("content", []))
        return max(1, chars // 4)

    if role == "bashExecution":
        return max(1, (len(str(msg.get("command", ""))) + len(str(msg.get("output", "")))) // 4)

    if role in ("compactionSummary", "branchSummary"):
        return max(1, len(str(msg.get("summary", ""))) // 4)

    return 1


def _get_last_usage(messages: list[Any]) -> tuple[dict, int] | None:
    """Find the last assistant message with a valid usage block."""
    for i in range(len(messages) - 1, -1, -1):
        msg = _unwrap(messages[i])
        if msg.get("role") != "assistant":
            continue
        usage = msg.get("usage", {})
        total = usage.get("total_tokens") or usage.get("total") or 0
        if total > 0 and msg.get("stop_reason") not in ("error", "aborted"):
            return usage, i
    return None


def calculate_context_tokens(messages: list[Any]) -> int:
    """
    Estimate context tokens using provider usage as anchor.
    More accurate than pure chars/4 summing.
    """
    usage_info = _get_last_usage(messages)
    if not usage_info:
        return sum(estimate_tokens(m) for m in messages)

    usage, idx = usage_info
    provider_tokens = (
        usage.get("total_tokens")
        or usage.get("total")
        or (usage.get("input", 0) + usage.get("output", 0))
    )
    trailing = sum(estimate_tokens(messages[i]) for i in range(idx + 1, len(messages)))
    return provider_tokens + trailing


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


# ── Cut Point Finding ──────────────────────────────────────────


def _find_valid_cut_points(entries: list[dict], start: int, end: int) -> list[int]:
    """Find positions where compaction can safely cut."""
    points = []
    for i in range(start, end):
        entry = entries[i]
        etype = entry.get("type", "")
        if etype == "message":
            role = entry.get("message", {}).get("role", "")
            if role in ("user", "assistant", "bashExecution", "custom", "branchSummary", "compactionSummary"):
                points.append(i)
        elif etype in ("branch_summary", "custom_message"):
            points.append(i)
    return points


def find_turn_start_index(entries: list[dict], entry_idx: int, start: int) -> int:
    """Find the user message that starts the turn containing entry_idx."""
    for i in range(entry_idx, start - 1, -1):
        entry = entries[i]
        etype = entry.get("type", "")
        if etype in ("branch_summary", "custom_message"):
            return i
        if etype == "message":
            role = entry.get("message", {}).get("role", "")
            if role in ("user", "bashExecution"):
                return i
    return -1


@dataclass
class CutPointResult:
    first_kept_entry_index: int = 0
    turn_start_index: int = -1
    is_split_turn: bool = False


def find_cut_point(
    entries: list[dict],
    start: int,
    end: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """Find the compaction cut point that keeps ~keepRecentTokens."""
    cut_points = _find_valid_cut_points(entries, start, end)
    if not cut_points:
        return CutPointResult(first_kept_entry_index=start)

    accumulated = 0
    cut_idx = cut_points[0]

    for i in range(end - 1, start - 1, -1):
        entry = entries[i]
        if entry.get("type") != "message":
            continue
        accumulated += estimate_tokens(entry.get("message", entry))
        if accumulated >= keep_recent_tokens:
            for cp in cut_points:
                if cp >= i:
                    cut_idx = cp
                    break
            break

    # Don't cut inside a compaction block
    while cut_idx > start:
        prev = entries[cut_idx - 1]
        if prev.get("type") in ("compaction", "message"):
            break
        cut_idx -= 1

    cut_entry = entries[cut_idx]
    is_user = cut_entry.get("type") == "message" and cut_entry.get("message", {}).get("role") == "user"
    turn_start = -1 if is_user else find_turn_start_index(entries, cut_idx, start)

    return CutPointResult(
        first_kept_entry_index=cut_idx,
        turn_start_index=turn_start,
        is_split_turn=not is_user and turn_start != -1,
    )


# ── Summarization Prompts ──────────────────────────────────────

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Read a conversation and produce "
    "a structured summary. Do NOT continue the conversation. Only output the summary."
)

SUMMARIZATION_PROMPT = """Create a structured context summary for an LLM to continue the work.

## Goal
[What is the user trying to accomplish?]

## Progress
### Done
- [x] [Completed tasks]

### In Progress
- [ ] [Current work]

## Key Decisions
- **[Decision]**: [Rationale]

## Next Steps
1. [Ordered next actions]

## Critical Context
- [File paths, function names, error messages, constraints — anything needed to continue]
"""

UPDATE_SUMMARIZATION_PROMPT = """The above is NEW conversation to merge into the existing summary in <previous-summary>.

RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, context
- UPDATE Progress: move "In Progress" to "Done" when completed
- UPDATE Next Steps based on current state
- PRESERVE exact file paths, function names, error messages

Use the SAME format as the previous summary.
"""

TURN_PREFIX_PROMPT = """This is the PREFIX of a turn. The SUFFIX (recent work) is retained.
Summarize the prefix to provide context:

## Original Request
[User's request for this turn]

## Early Progress
- [Key work done in prefix]

## Context for Suffix
- [What's needed to understand the kept suffix]
"""


# ── File Operations ────────────────────────────────────────────


@dataclass
class FileOperations:
    read: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def _extract_file_ops(messages: list[Any]) -> FileOperations:
    """Extract read/modified file paths from messages."""
    ops = FileOperations()
    for msg in messages:
        msg = _unwrap(msg)
        if msg.get("role") == "toolResult":
            tool_name = msg.get("tool_name", msg.get("toolName", ""))
            if tool_name == "read":
                # Read tool — extract path from content
                for b in msg.get("content", []):
                    text = b.get("text", "")
                    # Try to find file path in read result
            elif tool_name in ("write", "edit"):
                ops.edited.add(str(msg.get("details", {})))
        # Also check details
        details = msg.get("details", {})
        if isinstance(details, dict):
            path = details.get("path") or details.get("file_path") or details.get("filePath")
            role = msg.get("role", "")
            if role == "toolResult":
                ops.edited.add(str(path))
    return ops


def _format_file_ops(ops: FileOperations) -> str:
    parts = []
    if ops.read:
        parts.append("\n\n### Files Read\n" + "\n".join(f"- `{f}`" for f in sorted(ops.read)))
    if ops.edited:
        parts.append("\n\n### Files Modified\n" + "\n".join(f"- `{f}`" for f in sorted(ops.edited)))
    return "".join(parts)


# ── Summarization ──────────────────────────────────────────────


async def generate_summary(
    messages: list[Any],
    model: Model,
    *,
    api_key: str | None = None,
    reserve_tokens: int = 16384,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
    thinking_level: ThinkingLevel = "off",
) -> str:
    """Generate or update a structured conversation summary."""
    max_tokens = min(int(0.8 * reserve_tokens), model.max_tokens or 8192)

    # Choose prompt
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    # Serialize messages
    conversation_text = _serialize(messages)

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    context = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": time.time() * 1000,
        }],  # type: ignore[list-item]
    )

    reasoning = thinking_level if model.reasoning and thinking_level != "off" else None

    parts = []
    async for event in stream_simple(model, context, api_key=api_key, reasoning=reasoning):
        if event.type == "text_delta" and event.delta:
            parts.append(event.delta)
        elif event.type == "done" and event.message:
            texts = [b.get("text", "") for b in event.message.content if b.get("type") == "text"]
            return "\n".join(texts)

    return "".join(parts)


async def _generate_turn_prefix_summary(
    messages: list[Any],
    model: Model,
    api_key: str | None = None,
    reserve_tokens: int = 16384,
    thinking_level: ThinkingLevel = "off",
) -> str:
    """Summarize a split-turn prefix."""
    max_tokens = min(int(0.5 * reserve_tokens), model.max_tokens or 4096)

    conversation_text = _serialize(messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_PROMPT}"

    context = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": time.time() * 1000,
        }],  # type: ignore[list-item]
    )

    reasoning = thinking_level if model.reasoning and thinking_level != "off" else None

    parts = []
    async for event in stream_simple(model, context, api_key=api_key, reasoning=reasoning):
        if event.type == "text_delta" and event.delta:
            parts.append(event.delta)
        elif event.type == "done" and event.message:
            texts = [b.get("text", "") for b in event.message.content if b.get("type") == "text"]
            return "\n".join(texts)

    return "".join(parts)


# ── Main Compaction Pipeline ───────────────────────────────────


@dataclass
class CompactionPreparation:
    first_kept_entry_id: str = ""
    messages_to_summarize: list[Any] = field(default_factory=list)
    turn_prefix_messages: list[Any] = field(default_factory=list)
    is_split_turn: bool = False
    tokens_before: int = 0
    previous_summary: str | None = None
    file_ops: FileOperations = field(default_factory=FileOperations)


def prepare_compaction(
    entries: list[dict],
    settings: CompactionSettings,
) -> CompactionPreparation | None:
    """Prepare session entries for compaction. Returns None if not applicable."""
    if not entries or entries[-1].get("type") == "compaction":
        return None

    # Find previous compaction boundary
    prev_compaction_idx = -1
    previous_summary = None
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].get("type") == "compaction":
            prev_compaction_idx = i
            previous_summary = entries[i].get("summary", "")
            break

    boundary_start = 0
    if prev_compaction_idx >= 0:
        first_kept_id = entries[prev_compaction_idx].get("first_kept_entry_id", "")
        for i, e in enumerate(entries):
            if e.get("id") == first_kept_id:
                boundary_start = i
                break
        else:
            boundary_start = prev_compaction_idx + 1

    boundary_end = len(entries)

    # Calculate tokens
    messages = [e.get("message", e) for e in entries[boundary_start:boundary_end] if e.get("type") == "message"]
    tokens_before = calculate_context_tokens(messages)

    # Find cut point
    cut = find_cut_point(entries, boundary_start, boundary_end, settings.keep_recent_tokens)
    first_entry = entries[cut.first_kept_entry_index]
    first_kept_id = first_entry.get("id", "")
    if not first_kept_id:
        return None

    # Gather messages to summarize
    history_end = cut.turn_start_index if cut.is_split_turn else cut.first_kept_entry_index
    msgs_to_summarize = [
        e.get("message", e) for e in entries[boundary_start:history_end]
        if e.get("type") == "message" and e.get("message", {}).get("role") not in ("compaction", None)
    ]

    # Split-turn prefix
    turn_prefix = []
    if cut.is_split_turn:
        turn_prefix = [
            e.get("message", e) for e in entries[cut.turn_start_index:cut.first_kept_entry_index]
            if e.get("type") == "message"
        ]

    # File operations
    file_ops = _extract_file_ops(msgs_to_summarize)

    return CompactionPreparation(
        first_kept_entry_id=first_kept_id,
        messages_to_summarize=msgs_to_summarize,
        turn_prefix_messages=turn_prefix,
        is_split_turn=cut.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
    )


@dataclass
class CompactionResult:
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0
    details: dict = field(default_factory=dict)


async def compact(
    prep: CompactionPreparation,
    model: Model,
    *,
    api_key: str | None = None,
    custom_instructions: str | None = None,
    thinking_level: ThinkingLevel = "off",
) -> CompactionResult:
    """Run the full compaction."""
    settings = CompactionSettings()

    if prep.is_split_turn and prep.turn_prefix_messages:
        # Two-part summary: history + turn prefix
        if prep.messages_to_summarize:
            history = await generate_summary(
                prep.messages_to_summarize,
                model,
                api_key=api_key,
                reserve_tokens=settings.reserve_tokens,
                previous_summary=prep.previous_summary,
                custom_instructions=custom_instructions,
                thinking_level=thinking_level,
            )
        else:
            history = "No prior history."

        prefix = await _generate_turn_prefix_summary(
            prep.turn_prefix_messages,
            model,
            api_key=api_key,
            reserve_tokens=settings.reserve_tokens,
            thinking_level=thinking_level,
        )
        summary = f"{history}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix}"
    else:
        summary = await generate_summary(
            prep.messages_to_summarize,
            model,
            api_key=api_key,
            reserve_tokens=settings.reserve_tokens,
            previous_summary=prep.previous_summary,
            custom_instructions=custom_instructions,
            thinking_level=thinking_level,
        )

    summary += _format_file_ops(prep.file_ops)

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=prep.first_kept_entry_id,
        tokens_before=prep.tokens_before,
        details={
            "read_files": sorted(prep.file_ops.read),
            "modified_files": sorted(prep.file_ops.edited),
        },
    )


# ── Legacy API (kept for agent_harness.py compat) ──────────────


async def compact_conversation(
    messages: list[Any],
    model: Model,
    settings: CompactionSettings,
    context_window: int,
    api_key: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel = "off",
) -> CompactionResult | None:
    """Legacy wrapper — run full compaction on a flat message list."""
    tokens_before = calculate_context_tokens(messages)
    if not should_compact(tokens_before, context_window, settings):
        return None

    cut = find_cut_point_from_messages(messages, settings.keep_recent_tokens)
    if cut <= 0:
        return None

    to_summarize = messages[:cut]
    summary = await generate_summary(
        to_summarize, model, api_key=api_key,
        previous_summary=previous_summary, thinking_level=thinking_level,
    )

    return CompactionResult(
        summary=summary,
        tokens_before=tokens_before,
    )


def find_cut_point_from_messages(messages: list[Any], keep_recent: int) -> int:
    """Simple cut point for flat message lists."""
    accumulated = 0
    for i in range(len(messages) - 1, -1, -1):
        accumulated += estimate_tokens(messages[i])
        if accumulated >= keep_recent:
            return i + 1
    return 0


# ── Helpers ────────────────────────────────────────────────────


def _unwrap(msg: Any) -> dict:
    """Unwrap message object to dict."""
    if hasattr(msg, "__dict__"):
        return msg.__dict__
    if isinstance(msg, dict):
        return msg
    return {"role": "unknown", "content": str(msg)}


def _serialize(messages: list[Any]) -> str:
    """Convert messages to a compact text representation."""
    lines = []
    for msg in messages:
        msg = _unwrap(msg)
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if b.get("type") in ("text", None)
            )
        command = msg.get("command", "")
        if command:
            content = f"[bash: {command[:80]}] → {content[:200]}"
        lines.append(f"[{role}] {str(content)[:300]}")
    return "\n".join(lines)
