"""
Context window compaction — pi-mono parity.

Pipeline:
  estimate tokens (provider usage anchor) → should compact?
  → prepare (find cut point, gather entries, check split-turn)
  → summarize via LLM (full or incremental)
  → persist compaction entry (tree structure)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from connectclaw.injection import fold_ops, render_ops
from connectclaw.provider.stream import stream_simple
from connectclaw.provider.tokenizer import count_tokens, image_block_cost  # noqa: F401 (count_tokens re-exported for tests)
from connectclaw.provider.types import Context, Model, UserMessage

from ..types import AgentMessage, ThinkingLevel

# ── Token Estimation ───────────────────────────────────────────
#
# Text token counting lives in provider/tokenizer.py (tiktoken cl100k_base,
# CJK-aware offline fallback); image/image_ref blocks count at a fixed
# per-image cost via image_block_cost(). Kept here per-role so cut logic and
# budget math always have something realistic to accumulate.


# ── Settings ───────────────────────────────────────────────────


@dataclass
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16384
    # Trigger = (context_window - reserve_tokens) × trigger_ratio. The unused
    # 1-ratio is the buffer before the hard window: compact early, not when the
    # next tool output can already blow the request.
    trigger_ratio: float = 0.8
    keep_recent_tokens: int = 20000
    # Hard caps on what a compaction may put BACK into the context. Without them
    # the summary (incremental prompt said "preserve ALL previous information")
    # and the merged injection snapshot grow without bound: measured 13k + 16k
    # tokens in one real session, which pushed the post-compaction context back
    # over the trigger threshold and caused 88 compactions in a row.
    summary_max_tokens: int = 4096
    memory_snapshot_max_tokens: int = 2000


# ── Token Estimation ───────────────────────────────────────────


def estimate_tokens(message: Any) -> int:
    """Estimate token count per-message-type using the real tokenizer.

    Always returns >= 1 (callers accumulate/compare with this, so 0 would skew
    cut logic and downstream divisions)."""
    msg = _unwrap(message)
    role = msg.get("role", "")

    if role == "user":
        content = msg.get("content", "")
        if isinstance(content, str):
            return max(1, count_tokens(content))
        if isinstance(content, list):
            tokens = 0
            for b in content:
                if isinstance(b, dict) and b.get("type") in ("image", "image_ref"):
                    tokens += image_block_cost(b)
                else:
                    tokens += count_tokens(
                        str(b.get("text", "")) if isinstance(b, dict) else str(b)
                    )
            return max(1, tokens)
        return max(1, count_tokens(str(content)))

    if role == "assistant":
        content = msg.get("content", [])
        if isinstance(content, str):
            return max(1, count_tokens(content))
        tokens = 0
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("image", "image_ref"):
                tokens += image_block_cost(block)
                continue
            t = block.get("type", "") if isinstance(block, dict) else ""
            if t == "text":
                tokens += count_tokens(str(block.get("text", "")))
            elif t == "thinking":
                tokens += count_tokens(str(block.get("thinking", "")))
            elif t == "toolCall":
                tokens += count_tokens(str(block.get("name", "")) + str(block.get("arguments", {})))
            else:
                tokens += count_tokens(str(block))
        return max(1, tokens)

    if role == "toolResult":
        content = msg.get("content") or []
        tokens = 0
        for b in content:
            if isinstance(b, dict) and b.get("type") in ("image", "image_ref"):
                tokens += image_block_cost(b)
            else:
                tokens += count_tokens(
                    str(b.get("text", "")) if isinstance(b, dict) else str(b)
                )
        return max(1, tokens)

    if role == "bashExecution":
        return max(1, count_tokens(str(msg.get("command", "")) + str(msg.get("output", ""))))

    if role in ("compactionSummary", "branchSummary"):
        return max(1, count_tokens(str(msg.get("summary", ""))))

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


def compaction_trigger(context_window: int, settings: CompactionSettings) -> int:
    """Token count at which compaction starts.

    The usable budget is ``window - reserve`` (reserve covers output + system
    prompt/tools). Compacting at 80% of it — not at 100% — leaves a buffer so a
    long turn cannot blow through the window between two gate checks.
    """
    usable = max(1, context_window - settings.reserve_tokens)
    ratio = min(max(settings.trigger_ratio, 0.1), 1.0)
    return max(1, int(usable * ratio))


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    if not settings.enabled:
        return False
    return context_tokens > compaction_trigger(context_window, settings)


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
    "a structured summary. Do NOT continue the conversation. Only output the summary. "
    "Preserve exact file paths, commands, error strings, identifiers and numbers. "
    "You have NO tools available — never emit tool calls, function-call syntax, or "
    "special tokens; plain summary text only."
)

SUMMARIZATION_PROMPT = """Create a structured context summary for an LLM to continue the work.
Preserve exact file paths, commands, error strings, identifiers and numbers.
Do not invent progress; mark anything uncertain as uncertain.

## Goal
[What the user is trying to accomplish — the original request]

## Current State
[Where the work stands right now: done / running / blocked / waiting]

## Key Decisions
- **[Decision]**: [Rationale]

## Files & Code
- `path` — [what changed, or what matters about it]

## Errors & Fixes
- [Error string] → [cause] → [how it was fixed / still open]

## Constraints & Preferences
- [User requirements, environment limits, safety rules]

## Next Steps
1. [Ordered next actions]

## Critical Context
- [Exact paths, commands, ids, versions, numbers needed to continue]

BE CONCISE: this summary REPLACES the history it condenses. Keep only what is
needed to continue the work; drop resolved questions, obsolete attempts and
repeated detail. Aim for at most a few hundred lines.
"""

UPDATE_SUMMARIZATION_PROMPT = """The above is NEW conversation to merge into the existing summary in <previous-summary>.

RULES:
- REWRITE the whole summary in the same 8-section format; do not append a second summary
- KEEP what is still relevant from the previous summary — but DROP details that
  are resolved, obsolete or superseded (this summary replaces the history)
- ADD new progress, decisions, context
- MOVE completed items into Current State; keep Next Steps current
- PRESERVE exact file paths, commands, error strings, identifiers and numbers

BE CONCISE: merge instead of appending; never let the summary keep growing.
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


_EDIT_TOOLS = {"write", "edit", "hash_edit", "apply_patch", "patch"}
_READ_TOOLS = {"read", "hash_read"}


def _tool_call_files(messages: list[Any]) -> dict[str, tuple[str, str]]:
    """Map tool_call_id → (tool_name, path) from assistant tool-call blocks.

    Tool results do not carry the arguments they were called with, so the file
    list has to be reconstructed by pairing each result with its call. Without
    this, the summary's "Files" section only worked for tools that happened to
    put `path` in result details — i.e. almost never.
    """
    calls: dict[str, tuple[str, str]] = {}
    for msg in messages:
        msg = _unwrap(msg)
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "toolCall":
                continue
            args = block.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                continue
            path = (
                args.get("path") or args.get("file_path") or args.get("filePath")
            )
            if isinstance(path, str) and path.strip():
                calls[str(block.get("id", ""))] = (
                    str(block.get("name", "")),
                    path.strip(),
                )
    return calls


def _extract_file_ops(messages: list[Any]) -> FileOperations:
    """Extract read/modified file paths from messages.

    Two sources, in order: `details.path` on the result (legacy/attach_image),
    then the paired assistant tool call arguments (the normal path for
    read/hash_read/write/hash_edit, whose results do not persist arguments).
    """
    ops = FileOperations()
    calls = _tool_call_files(messages)
    for msg in messages:
        msg = _unwrap(msg)
        if msg.get("role") != "toolResult":
            continue
        details = msg.get("details", {})
        path = ""
        if isinstance(details, dict):
            path = (
                details.get("path")
                or details.get("file_path")
                or details.get("filePath")
                or ""
            )
        tool_name = str(msg.get("tool_name", msg.get("toolName", "")))
        if not path:
            call_name, call_path = calls.get(str(msg.get("tool_call_id", "")), ("", ""))
            tool_name = tool_name or call_name
            path = call_path
        if not (isinstance(path, str) and path.strip()):
            continue
        path = path.strip()[:_MAX_FILE_PATH_CHARS]
        if tool_name in _EDIT_TOOLS:
            ops.edited.add(path)
        elif tool_name in _READ_TOOLS:
            ops.read.add(path)
    return ops


_MAX_FILES_LISTED = 100
_MAX_FILE_PATH_CHARS = 200


def _format_file_ops(ops: FileOperations) -> str:
    """Render read/modified file lists, deduped and bounded.

    The naive version accumulated every edited path across every compaction
    forever (an unbounded struct that grows the summary each cycle). We now cap
    both the number of files and the length of each path.
    """
    parts = []
    for label, files in (("Read", ops.read), ("Modified", ops.edited)):
        if not files:
            continue
        fs = sorted(files)
        extra = ""
        if len(fs) > _MAX_FILES_LISTED:
            extra = f"\n- ... and {len(fs) - _MAX_FILES_LISTED} more"
            fs = fs[:_MAX_FILES_LISTED]
        body = "\n".join(f"- `{f[:_MAX_FILE_PATH_CHARS]}`" for f in fs)
        parts.append(f"\n\n### Files {label}\n{body}{extra}")
    return "".join(parts)


def _cap_summary(summary: str, max_tokens: int) -> str:
    """Hard cap on the final summary string (LLM text + file ops).

    The LLM portion is already bounded by `max_tokens` at generation time; this
    guard catches the assembled result (e.g. the file-ops tail) so the summary
    can never silently outgrow the keep-recent budget.
    """
    if not summary or max_tokens <= 0:
        return summary
    est = count_tokens(summary)
    if est <= max_tokens:
        return summary
    char_budget = max(1, int(len(summary) * max_tokens / max(1, est)))
    head = summary[:char_budget].rstrip()
    return f"{head}\n[summary truncated: {est}tk > budget {max_tokens}tk]\n"


# ── Summarization ──────────────────────────────────────────────


_log = logging.getLogger(__name__)

# 总结模型声明了无工具，但工具特训过的模型偶尔仍会吐原生调用语法；
# 无工具声明时 provider 不会解析这些特殊 token，原样漏进文本。
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<(?:\w+_)?(?:function_call|tool_call)\b[^>]*>.*?(?:</(?:tool_calls?|function_call)>|$)",
    re.DOTALL,
)


def strip_tool_call_artifacts(text: str) -> str:
    """Remove leaked native tool-call blocks (e.g. `<dots_function_call>…</tool_calls>`)."""
    return _TOOL_CALL_BLOCK_RE.sub("", text or "").strip()


async def generate_summary(
    messages: list[Any],
    model: Model,
    *,
    api_key: str | None = None,
    reserve_tokens: int = 16384,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
    thinking_level: ThinkingLevel = "off",
    signal: asyncio.Event | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate or update a structured conversation summary.

    ``max_tokens`` is a HARD output cap sent to the API. It used to be computed
    (0.8 × reserve ≈ 13k) and then silently dropped, so summaries could grow to
    the full model output limit every cycle.
    """
    max_tokens = max_tokens or min(int(0.8 * reserve_tokens), model.max_tokens or 8192)
    max_tokens = min(max_tokens, model.max_tokens or max_tokens)

    # Choose prompt
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    # Serialize messages (budget-capped so this summarizer call cannot itself
    # overflow the model window, and capped per-message via _serialize)
    conversation_text = _serialize(messages, max_tokens=_summarize_input_budget(model, reserve_tokens))

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    context = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[UserMessage(
            content=[{"type": "text", "text": prompt_text}],
            timestamp=time.time() * 1000,
        )],
    )

    reasoning = thinking_level if model.reasoning and thinking_level != "off" else None

    parts = []
    # M10：透传 signal——摘要 LLM 调用此前无法被 /stop 中断
    async for event in stream_simple(
        model, context, api_key=api_key, reasoning=reasoning,
        signal=signal, max_tokens=max_tokens,
    ):
        if event.type == "text_delta" and event.delta:
            parts.append(event.delta)
        elif event.type == "done" and event.message:
            texts = [b.get("text", "") for b in event.message.content if b.get("type") == "text"]
            return _finish_summary(messages, "\n".join(texts), model)

    return _finish_summary(messages, "".join(parts), model)


def _finish_summary(messages: list[Any], text: str, model: Model) -> str:
    """Sanitize summarizer output; on empty result fall back to a verbatim digest."""
    text = strip_tool_call_artifacts(text)
    if text:
        return text
    _log.warning("summarizer returned no usable text (model=%s); falling back to verbatim digest",
                 getattr(model, "id", "?"))
    digest = _serialize(messages, max_tokens=400).strip()
    return f"[auto-summary failed — most recent messages verbatim]\n{digest}" if digest else ""


async def _generate_turn_prefix_summary(
    messages: list[Any],
    model: Model,
    api_key: str | None = None,
    reserve_tokens: int = 16384,
    thinking_level: ThinkingLevel = "off",
    signal: asyncio.Event | None = None,
    max_tokens: int | None = None,
) -> str:
    """Summarize a split-turn prefix."""
    max_tokens = max_tokens or min(int(0.5 * reserve_tokens), model.max_tokens or 4096)
    max_tokens = min(max_tokens, model.max_tokens or max_tokens)

    conversation_text = _serialize(messages, max_tokens=_summarize_input_budget(model, reserve_tokens))
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_PROMPT}"

    context = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[UserMessage(
            content=[{"type": "text", "text": prompt_text}],
            timestamp=time.time() * 1000,
        )],
    )

    reasoning = thinking_level if model.reasoning and thinking_level != "off" else None

    parts = []
    async for event in stream_simple(
        model, context, api_key=api_key, reasoning=reasoning,
        signal=signal, max_tokens=max_tokens,
    ):
        if event.type == "text_delta" and event.delta:
            parts.append(event.delta)
        elif event.type == "done" and event.message:
            texts = [b.get("text", "") for b in event.message.content if b.get("type") == "text"]
            return strip_tool_call_artifacts("\n".join(texts))

    return strip_tool_call_artifacts("".join(parts))


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
    # 区域内注入块 squash 成的"当前状态"（见 connectclaw/injection.py）
    merged_context: str = ""
    context_state: dict | None = None
    # ── Hardening metadata ──
    liberated_tokens: int = 0          # estimated tokens condensed into summary
    post_compact_estimate: int = 0     # estimated kept + summary tokens after compaction
    fits_budget: bool = True           # whether post-compaction lands below the trigger


def prepare_compaction(
    entries: list[dict],
    settings: CompactionSettings,
    context_window: int | None = None,
) -> CompactionPreparation | None:
    """Prepare session entries for compaction. Returns None if not applicable.

    Hardening vs. the naive version:
      * **Benefit guard**: if there is nothing to condense (the cut marker would
        keep everything — the degenerate "add a summary, shrink nothing" case),
        return None instead of appending a pointless compaction entry that only
        grows the JSONL and re-triggers on every turn.
      * **Budget fit**: when `context_window` is given, shrink the keep-recent
        window until post-compaction context (kept messages + new summary) fits
        below the trigger line. This is the direct livelock breaker: the old
        code kept ~`keep_recent_tokens` by a chars/4 estimate that under-counted
        CJK 2-4x, so the window alone could exceed the budget and compaction
        never actually shrinks the context.

    The search is bounded (at most 8 cut-point scans, no LLM calls), and the
    recent window never drops below `max(2048, keep_recent // 4)`.
    """
    if not entries or entries[-1].get("type") == "compaction":
        return None

    # Find previous compaction boundary
    prev_compaction_idx = -1
    previous_summary = None
    previous_snapshot: dict | None = None
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].get("type") == "compaction":
            prev_compaction_idx = i
            previous_summary = entries[i].get("summary", "")
            # 上一轮压缩 squash 出来的状态块：它是当前上下文的一部分，而这次压缩
            # 正要把它替换掉，所以必须并进新的合并结果里。
            previous_snapshot = entries[i].get("context_state") or None
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

    # Current context of the compactible window (usage-anchored when available)
    window_msgs = [
        e.get("message", e)
        for e in entries[boundary_start:boundary_end]
        if e.get("type") == "message"
    ]
    window_tokens = calculate_context_tokens(window_msgs)

    # The post-compaction target is below the TRIGGER (not below the hard
    # window): landing just under the trigger is exactly what caused the
    # 88-compaction livelock. `budget` here means "the largest post-compaction
    # size we accept", so it is the trigger line.
    trigger = compaction_trigger(context_window, settings) if context_window else None
    budget = trigger
    floor = max(2048, int(settings.keep_recent_tokens * 0.25))
    # Summary output is hard-capped at generation time now (summary_max_tokens);
    # when the knob is 0/misconfigured fall back to a conservative quarter-reserve.
    summary_cap = settings.summary_max_tokens
    if summary_cap <= 0:
        summary_cap = max(2048, int(settings.reserve_tokens * 0.25))
    # Post-compaction must land BELOW the trigger with room to spare, otherwise
    # the very next request compacts again (the observed 88-compaction livelock).
    headroom = max(2048, int((context_window or 0) * 0.05))
    target = (budget - headroom) if budget is not None else None

    # The usage anchor already counted injection blocks and the previous
    # summary (they were part of that request). Only subtract those when an
    # anchor exists — the raw-sum fallback never included them.
    anchored = _get_last_usage(window_msgs) is not None
    prev_summary_tokens = count_tokens(previous_summary) if (anchored and previous_summary) else 0

    keep = settings.keep_recent_tokens
    chosen: tuple[CutPointResult, list[Any], int, int, str, dict | None] | None = None
    #                        cut, msgs, freed, post, merged_context, context_state

    for _ in range(8):
        cut = find_cut_point(entries, boundary_start, boundary_end, keep)
        first_entry = entries[cut.first_kept_entry_index]
        first_kept_id = first_entry.get("id", "")
        if not first_kept_id:
            return None

        history_end = cut.turn_start_index if cut.is_split_turn else cut.first_kept_entry_index
        msgs = [
            e.get("message", e) for e in entries[boundary_start:history_end]
            if e.get("type") == "message" and e.get("message", {}).get("role") not in ("compaction", None)
        ]

        if not msgs:
            # Degenerate: nothing to condense at this keep-width. With a budget we
            # can dig deeper (a smaller keep window will cut real messages);
            # otherwise there is genuinely nothing useful to do.
            if budget is None or keep <= floor:
                return None
            keep = max(floor, int(keep * 0.6))
            continue

        freed = sum(estimate_tokens(m) for m in msgs)
        merged_context, merged_ops, old_injection = _squash_region(
            entries, boundary_start, history_end, previous_snapshot,
            max_memory_tokens=settings.memory_snapshot_max_tokens,
        )
        merged_tokens = count_tokens(merged_context) if merged_context else 0
        summary_est = min(summary_cap, freed)
        # What survives: kept messages + new summary + merged injection state.
        # `old_injection`/`prev_summary_tokens` were counted by the usage anchor
        # and are being replaced by `merged_tokens`/`summary_est`.
        replaced = freed + (old_injection + prev_summary_tokens if anchored else 0)
        post_est = max(0, window_tokens - replaced) + summary_est + merged_tokens
        chosen = (cut, msgs, freed, post_est, merged_context, merged_ops[0] if merged_ops else None)
        if target is None or post_est <= target or keep <= floor:
            break
        keep = max(floor, int(keep * 0.6))

    if chosen is None:
        return None
    cut, msgs_to_summarize, freed, post_est, merged_context, context_state = chosen

    # Shrink guard (DSH-style): never persist a compaction that barely moves the
    # needle. Otherwise a large kept-window or summary can leave the context at
    # or above its previous size while still paying for a full summarization —
    # the "compact → still over → compact again" livelock.
    min_gain = max(1024, int(window_tokens * 0.01))
    if window_tokens - post_est <= min_gain:
        _log.info(
            "compaction skipped: estimated gain %d tok <= floor %d "
            "(before=%d, after=%d)",
            window_tokens - post_est, min_gain, window_tokens, post_est,
        )
        return None

    first_kept_id = entries[cut.first_kept_entry_index].get("id", "")

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
        tokens_before=window_tokens,
        previous_summary=previous_summary,
        file_ops=file_ops,
        liberated_tokens=freed,
        post_compact_estimate=post_est,
        fits_budget=(budget is None) or (post_est <= budget),
        merged_context=merged_context,
        context_state=context_state,
    )


def _squash_region(
    entries: list[dict],
    boundary_start: int,
    history_end: int,
    previous_snapshot: dict | None,
    *,
    max_memory_tokens: int,
) -> tuple[str, list[dict], int]:
    """Fold the region's injection ops into ONE current state (git-merge style).

    返回 ``(merged_context, snapshot_ops, old_injection_tokens)``。第三条是这批
    context 条目**当初**在请求里占的 token——预算估算要从 usage 锚点里扣掉它，
    否则会重复计算、把压缩目标算得过于乐观/悲观。
    """
    region_batches: list[list[dict]] = []
    old_tokens = 0
    for e in entries[boundary_start:history_end]:
        if e.get("type") != "context":
            continue
        ops = list(e.get("ops") or [])
        if ops:
            region_batches.append(ops)
            old_tokens += count_tokens(render_ops(ops))
    # 上一个压缩的快照也要并进来——它代表这次压缩正要替换掉的那段历史。
    if previous_snapshot:
        region_batches.insert(0, [previous_snapshot])
        old_tokens += count_tokens(render_ops([previous_snapshot]))

    state = fold_ops(region_batches)
    merged_ops = state.snapshot_ops(max_tokens=max_memory_tokens)
    merged_context = state.render(merged_ops, incremental=False)
    return merged_context, merged_ops, old_tokens


@dataclass
class CompactionResult:
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0
    details: dict = field(default_factory=dict)
    # 被压缩区域内散落的注入块 squash 成的"当前状态"（git 合并语义）
    merged_context: str = ""
    context_state: dict | None = None


def _summarize_input_budget(model: Model, reserve_tokens: int) -> int:
    """Cap `<conversation>` tokens sent to the summarizer so the summarizer's own
    call stays well inside the model window (input + output + prompt overhead)."""
    window = getattr(model, "context_window", 0) or 65536
    out = min(int(0.8 * reserve_tokens), model.max_tokens or 8192)
    overhead = int(reserve_tokens * 0.5)
    return max(4096, window - out - overhead)


async def compact(
    prep: CompactionPreparation,
    model: Model,
    *,
    api_key: str | None = None,
    custom_instructions: str | None = None,
    thinking_level: ThinkingLevel = "off",
    settings: CompactionSettings | None = None,
    signal: asyncio.Event | None = None,
) -> CompactionResult:
    """Run the full compaction."""
    settings = settings or CompactionSettings()

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
                signal=signal,
                max_tokens=settings.summary_max_tokens,
            )
        else:
            history = "No prior history."

        prefix = await _generate_turn_prefix_summary(
            prep.turn_prefix_messages,
            model,
            api_key=api_key,
            reserve_tokens=settings.reserve_tokens,
            thinking_level=thinking_level,
            signal=signal,
            max_tokens=max(1024, settings.summary_max_tokens // 2),
        )
        if prefix:
            summary = f"{history}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix}"
        else:
            # 保留区本就带着 suffix 原文，前缀摘要失效时宁缺毋滥
            _log.warning("split-turn prefix summary empty; omitting Turn Context section")
            summary = history
    else:
        summary = await generate_summary(
            prep.messages_to_summarize,
            model,
            api_key=api_key,
            reserve_tokens=settings.reserve_tokens,
            previous_summary=prep.previous_summary,
            custom_instructions=custom_instructions,
            thinking_level=thinking_level,
            signal=signal,
            max_tokens=settings.summary_max_tokens,
        )

    summary += _format_file_ops(prep.file_ops)
    # 二次硬封顶：文件清单尾巴 + split-turn 拼接都可能超出 API 输出上限。
    summary = _cap_summary(summary, settings.summary_max_tokens)

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=prep.first_kept_entry_id,
        tokens_before=prep.tokens_before,
        details={
            "read_files": sorted(prep.file_ops.read),
            "modified_files": sorted(prep.file_ops.edited),
            "liberated_tokens": prep.liberated_tokens,
            "post_compact_estimate": prep.post_compact_estimate,
            "fits_budget": prep.fits_budget,
        },
        merged_context=prep.merged_context,
        context_state=prep.context_state,
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
    signal: asyncio.Event | None = None,
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
        signal=signal,
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


_SUMMARY_MSG_CHARS = {
    "toolResult": 8000,
    "assistant": 4000,
    "user": 4000,
    "bashExecution": 4000,
}
_SUMMARY_DEFAULT_CHARS = 2000
_SUMMARY_THINKING_CHARS = 2000
_SUMMARY_TOOL_ARGS_CHARS = 1000


def _clip_for_summary(text: str, limit: int) -> str:
    """Head+tail clip so the summarizer keeps both the setup and the result."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.75)
    tail = max(0, limit - head)
    omitted = len(text) - limit
    return f"{text[:head]}\n…[{omitted} chars omitted]…\n{text[-tail:]}"


def _serialize(messages: list[Any], max_tokens: int | None = None) -> str:
    """Convert messages to a compact text representation.

    Per-message clipping is role-aware and head+tail (not the old flat 300-char
    head cut): tool results keep 8k chars, so summaries retain command output
    and file excerpts instead of a useless opening fragment. Thinking and tool
    calls are included because they carry the reasoning and the exact actions.

    When `max_tokens` is given and the serialized size still exceeds it, the
    OLDEST messages are dropped (newest preserved) with a note, so the
    summarizer input stays bounded even for very long sessions.
    """
    lines = []
    for msg in messages:
        msg = _unwrap(msg)
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            images = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype in ("text", None):
                    parts.append(str(b.get("text", "")))
                elif btype == "thinking":
                    parts.append(
                        "[thinking] "
                        + _clip_for_summary(str(b.get("thinking", "")), _SUMMARY_THINKING_CHARS)
                    )
                elif btype == "toolCall":
                    args = b.get("arguments", {})
                    if not isinstance(args, str):
                        try:
                            args = json.dumps(args, ensure_ascii=False)
                        except Exception:
                            args = str(args)
                    parts.append(
                        f"[toolCall {b.get('name', '')} "
                        f"{_clip_for_summary(args, _SUMMARY_TOOL_ARGS_CHARS)}]"
                    )
                elif btype in ("image", "image_ref"):
                    images.append(b)
            content = " ".join(p for p in parts if p)
            if images:
                content += " " + " ".join(
                    f"[image {b.get('id', '')} {b.get('mime_type', '')} {int(b.get('size', 0)) // 1024}KB]"
                    for b in images
                )
        command = msg.get("command", "")
        if command:
            out = _clip_for_summary(str(content), _SUMMARY_MSG_CHARS["bashExecution"])
            content = f"[bash: {command[:80]}] → {out}"
        else:
            content = _clip_for_summary(
                str(content), _SUMMARY_MSG_CHARS.get(role, _SUMMARY_DEFAULT_CHARS)
            )
        lines.append(f"[{role}] {content}")

    if max_tokens and max_tokens > 0:
        counts = [count_tokens(line) for line in lines]
        total = sum(counts)
        if total > max_tokens:
            kept: list[str] = []
            acc = 0
            for line, c in zip(reversed(lines), reversed(counts)):
                if kept and acc + c > max_tokens:
                    break
                kept.append(line)
                acc += c
            kept.reverse()
            dropped = len(lines) - len(kept)
            if dropped:
                lines = [f"[note] oldest {dropped} messages dropped to fit {max_tokens}tk summarize input"] + kept

    return "\n".join(lines)
