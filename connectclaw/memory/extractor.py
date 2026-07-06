"""Automatic memory extraction from conversations."""

from __future__ import annotations

import json
import time
from typing import Any

from connectclaw.logging import get_logger
from connectclaw.provider.stream import stream_simple
from connectclaw.provider.types import Context, Model, UserMessage

from .prompts import EXTRACTION_PROMPT, EXTRACTION_SYSTEM_PROMPT
from .types import MemoryEntry, MemoryType

logger = get_logger(__name__)


async def extract_memories(
    messages: list[dict[str, Any]],
    model: Model,
    *,
    api_key: str | None = None,
    existing_memories: list[MemoryEntry] | None = None,
    source_session: str | None = None,
) -> list[MemoryEntry]:
    """Extract memorable information from a conversation.

    Runs an LLM call to analyze the conversation and return structured memory entries.
    Does NOT persist — caller decides what to do with the results.
    """
    conversation_text = _serialize_messages(messages)
    if len(conversation_text) < 100:
        return []

    existing_text = _format_existing(existing_memories or [])

    prompt_text = EXTRACTION_PROMPT.format(
        conversation=conversation_text[:8000],
        existing_memories=existing_text[:3000],
    )

    context = Context(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        messages=[UserMessage(content=prompt_text, timestamp=time.time() * 1000)],
    )

    text = await _call_llm(context, model, api_key=api_key)
    if not text:
        return []

    entries = _parse_extraction_result(text, source_session=source_session)
    logger.info("Extracted %d memories from conversation", len(entries))
    return entries


def _serialize_messages(messages: list[dict[str, Any]]) -> str:
    """Convert messages to a compact text for LLM analysis."""
    lines = []
    for msg in messages:
        msg_dict = _unwrap(msg)
        role = msg_dict.get("role", "?")
        content = msg_dict.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        command = msg_dict.get("command", "")
        tool_name = msg_dict.get("tool_name", "")
        if command:
            content = f"[bash: {command[:120]}] -> {str(content)[:300]}"
        elif tool_name:
            content = f"[{tool_name}] {str(content)[:300]}"
        else:
            content = str(content)[:400]
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _unwrap(msg: Any) -> dict:
    if hasattr(msg, "__dict__"):
        return {k: v for k, v in msg.__dict__.items() if not k.startswith("_")}
    if isinstance(msg, dict):
        return msg
    return {"role": "unknown", "content": str(msg)}


def _format_existing(memories: list[MemoryEntry]) -> str:
    if not memories:
        return "(no existing memories)"
    lines = []
    for m in memories[:50]:
        lines.append(f"- [{m.type.value}] {m.content}")
    return "\n".join(lines)


def _parse_extraction_result(
    text: str, *, source_session: str | None = None
) -> list[MemoryEntry]:
    """Parse LLM output into MemoryEntry objects."""
    text = text.strip()

    if "```" in text:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                items = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning("Failed to parse extraction result: %s", text[:200])
                return []
        else:
            return []

    if not isinstance(items, list):
        return []

    now = time.time()
    entries = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            mem_type = MemoryType(item.get("type", "semantic"))
        except ValueError:
            mem_type = MemoryType.SEMANTIC

        content = item.get("content", "").strip()
        if not content:
            continue

        entries.append(
            MemoryEntry(
                type=mem_type,
                content=content,
                detail=item.get("detail"),
                category=item.get("category", ""),
                importance=min(1.0, max(0.0, float(item.get("importance", 0.5)))),
                created_at=now,
                last_accessed=now,
                source_session=source_session,
            )
        )

    return entries


async def _call_llm(
    context: Context,
    model: Model,
    *,
    api_key: str | None = None,
) -> str:
    """Make a simple LLM call and return the text response."""
    parts: list[str] = []
    final: str | None = None
    async for event in stream_simple(model, context, api_key=api_key):
        if event.type == "text_delta" and event.delta:
            parts.append(event.delta)
        elif event.type == "done" and event.message:
            texts = [
                b.get("text", "")
                for b in event.message.content
                if b.get("type") == "text"
            ]
            final = "\n".join(texts)
            # break (not return) so the stream generator closes cleanly while
            # the event loop is still alive — avoids GeneratorExit at shutdown.
            break
    return final if final is not None else "".join(parts)
