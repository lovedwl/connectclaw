"""Async streaming over OpenAI-compatible API."""

import asyncio
import time
from typing import Any, AsyncIterator

from connectclaw.logging import get_logger

from .deepseek import DeepSeekProvider
from .types import (
    AssistantMessage,
    Context,
    Model,
    StreamEvent,
)

logger = get_logger(__name__)

# Default provider instance
_provider = DeepSeekProvider()

# Client cache — reuse AsyncOpenAI clients across calls.
# Key: (base_url, api_key_first_8), Value: AsyncOpenAI
_client_cache: dict[tuple[str, str], Any] = {}
_client_cache_lock = asyncio.Lock()


async def stream_simple(
    model: Model,
    context: Context,
    *,
    api_key: str | None = None,
    signal: asyncio.Event | None = None,
    reasoning: str | None = None,
    session_id: str | None = None,
    timeout_ms: int = 300_000,
    max_retries: int = 3,
    base_url: str | None = None,
    _provider_instance: DeepSeekProvider | None = None,
) -> AsyncIterator[StreamEvent]:
    """
    Async generator yielding StreamEvents from LLM API.

    Contract (mirrors pi-mono):
    - Never raises for model/request failures.
    - Encodes failures as `{type: "error", ...}` or stop_reason="error".
    - Supports cancellation via `signal` (asyncio.Event).

    Usage:
        async for event in stream_simple(model, context, api_key="sk-..."):
            match event.type:
                case "text_delta": ...
                case "thinking_delta": ...
                case "toolcall_delta": ...
                case "done": ...
                case "error": ...
    """
    provider = _provider_instance or _provider
    base = base_url or model.base_url
    key = api_key or ""

    # Reuse cached client — creating a new AsyncOpenAI per call leaks
    # httpx.AsyncClient connection pools (each ~5-10 MB).
    cache_key = (base, key[:8] if key else "")
    async with _client_cache_lock:
        if cache_key in _client_cache:
            client = _client_cache[cache_key]
        else:
            client = provider.build_client(key, base_url=base)
            _client_cache[cache_key] = client

    # Build the initial partial message
    partial = AssistantMessage(
        content=[],
        model=model.id,
        stop_reason="stop",
        usage={},
        timestamp=time.time() * 1000,
    )

    # Track tool calls separately from content blocks.
    # The API's tc_delta.index is the position within the tool_calls list,
    # NOT within the overall content array — conflating them corrupts
    # thinking/text blocks.
    _tool_call_slots: dict[int, dict] = {}

    # Emit start event
    yield StreamEvent(type="start", partial=partial)

    # Build request params
    messages = provider.convert_messages(context.messages)
    if context.system_prompt:
        messages.insert(0, {"role": "system", "content": context.system_prompt})

    tools = provider.convert_tools(context.tools)

    params: dict = {
        "model": model.id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        params["tools"] = tools

    # Reasoning effort (thinking mode)
    if reasoning and reasoning != "off":
        params["extra_body"] = {"reasoning_effort": reasoning}

    timeout = timeout_ms / 1000.0

    logger.debug("stream_simple: model=%s base_url=%s messages=%d tools=%d reasoning=%s",
                 model.id, base_url or model.base_url, len(messages), len(tools) if tools else 0, reasoning)

    for attempt in range(max_retries):
        try:
            stream = await client.chat.completions.create(**params, timeout=timeout)

            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                # Check cancellation
                if signal and signal.is_set():
                    partial.stop_reason = "aborted"
                    partial.error_message = "Request aborted"
                    yield StreamEvent(
                        type="error",
                        error_message="Request aborted",
                        message=partial,
                    )
                    return

                # Skip chunks without choices
                if not chunk.choices:
                    # Final chunk with usage info
                    if hasattr(chunk, "usage") and chunk.usage:
                        partial.usage = {
                            "input": chunk.usage.prompt_tokens or 0,
                            "output": chunk.usage.completion_tokens or 0,
                            "total": chunk.usage.total_tokens or 0,
                        }
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Log first few content-carrying chunks
                if chunk_count <= 3:
                    has_reasoning = hasattr(delta, "reasoning_content") and delta.reasoning_content
                    has_content = bool(delta.content)
                    has_tool_calls = bool(delta.tool_calls)
                    logger.debug("stream chunk[%d]: reasoning=%s text=%s tool_calls=%s finish=%s",
                                 chunk_count, has_reasoning, has_content, has_tool_calls, finish_reason)

                # Handle reasoning_content (thinking mode)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_text = delta.reasoning_content
                    # Ensure thinking block exists
                    if not partial.content or partial.content[-1].get("type") != "thinking":
                        partial.content.append({"type": "thinking", "thinking": ""})
                    idx = len(partial.content) - 1
                    partial.content[idx]["thinking"] = partial.content[idx]["thinking"] + reasoning_text
                    yield StreamEvent(
                        type="thinking_delta",
                        delta=reasoning_text,
                        content_index=idx,
                        partial=partial,
                    )

                # Handle text content
                if delta.content:
                    text = delta.content
                    if not partial.content or partial.content[-1].get("type") != "text":
                        partial.content.append({"type": "text", "text": ""})
                    idx = len(partial.content) - 1
                    partial.content[idx]["text"] = partial.content[idx]["text"] + text
                    yield StreamEvent(
                        type="text_delta",
                        delta=text,
                        content_index=idx,
                        partial=partial,
                    )

                # Handle tool calls — accumulate in separate dict keyed by API index
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        api_idx = tc_delta.index
                        slot = _tool_call_slots.get(api_idx)
                        if slot is None:
                            slot = {
                                "type": "toolCall",
                                "id": "",
                                "name": "",
                                "arguments": {},
                                "_args_json": "",
                            }
                            _tool_call_slots[api_idx] = slot
                        if tc_delta.id:
                            slot["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                slot["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                slot["_args_json"] = slot.get("_args_json", "") + tc_delta.function.arguments

                # Handle finish — parse accumulated tool call args + set stop reason
                if finish_reason:
                    # Merge tool call slots (appended after thinking/text blocks)
                    if _tool_call_slots:
                        import json as _json
                        for api_idx in sorted(_tool_call_slots):
                            slot = _tool_call_slots[api_idx]
                            if slot.get("_args_json"):
                                try:
                                    slot["arguments"] = _json.loads(slot.pop("_args_json"))
                                except _json.JSONDecodeError:
                                    slot["arguments"] = {}
                                    slot.pop("_args_json", None)
                            else:
                                slot.pop("_args_json", None)
                            partial.content.append(slot)
                        _tool_call_slots.clear()

                    match finish_reason:
                        case "stop":
                            partial.stop_reason = "stop"
                        case "length":
                            partial.stop_reason = "length"
                        case "tool_calls":
                            partial.stop_reason = "toolUse"

                    logger.debug("stream done: stop_reason=%s content_blocks=%d chunks=%d",
                                 partial.stop_reason, len(partial.content), chunk_count)
                    for i, b in enumerate(partial.content):
                        logger.debug("  block[%d]: type=%s text_len=%d thinking_len=%d",
                                     i, b.get("type", "?"),
                                     len(b.get("text", "")),
                                     len(b.get("thinking", "")))
                    yield StreamEvent(
                        type="done",
                        message=partial,
                    )
                    return

            # Stream ended without explicit finish_reason
            if _tool_call_slots:
                import json as _json
                for api_idx in sorted(_tool_call_slots):
                    slot = _tool_call_slots[api_idx]
                    if slot.get("_args_json"):
                        try:
                            slot["arguments"] = _json.loads(slot.pop("_args_json"))
                        except _json.JSONDecodeError:
                            slot["arguments"] = {}
                            slot.pop("_args_json", None)
                    else:
                        slot.pop("_args_json", None)
                    partial.content.append(slot)
                _tool_call_slots.clear()
            logger.debug("stream ended without finish_reason: chunks=%d content_blocks=%d",
                         chunk_count, len(partial.content))
            for i, b in enumerate(partial.content):
                logger.debug("  block[%d]: type=%s text_len=%d thinking_len=%d",
                             i, b.get("type", "?"),
                             len(b.get("text", "")),
                             len(b.get("thinking", "")))
            yield StreamEvent(type="done", message=partial)
            return

        except asyncio.CancelledError:
            partial.stop_reason = "aborted"
            partial.error_message = "Request cancelled"
            yield StreamEvent(
                type="error",
                error_message="Request cancelled",
                message=partial,
            )
            return

        except Exception as e:
            error_msg = str(e)
            logger.debug("stream error (attempt %d/%d): %s", attempt + 1, max_retries, error_msg[:200])
            # Check for retryable errors
            if _is_retryable(error_msg) and attempt < max_retries - 1:
                delay = min(2**attempt, 30)
                await asyncio.sleep(delay)
                continue

            partial.stop_reason = "error"
            partial.error_message = error_msg
            yield StreamEvent(
                type="error",
                error_message=error_msg,
                message=partial,
            )
            return


def _is_retryable(error_msg: str) -> bool:
    """Check if an error is retryable."""
    retryable = [
        "rate_limit",
        "rate limit",
        "too many requests",
        "429",
        "timeout",
        "connection",
        "server_error",
        "500",
        "502",
        "503",
        "busy",
    ]
    msg_lower = error_msg.lower()
    return any(pattern in msg_lower for pattern in retryable)
