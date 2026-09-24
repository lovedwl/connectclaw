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
# Each client owns an httpx connection pool (~5-10 MB), so the cache is bounded
# and the oldest client is closed when a config rotation pushes past the cap.
_client_cache_max = 8


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
    proxy: str | None = None,
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
    proxy = proxy or getattr(model, "proxy", "") or None

    # Reuse cached client — creating a new AsyncOpenAI per call leaks
    # httpx.AsyncClient connection pools (each ~5-10 MB).
    # Proxy is part of the key so switching it creates a fresh client.
    cache_key = (base, key[:8] if key else "", proxy or "")
    async with _client_cache_lock:
        if cache_key in _client_cache:
            client = _client_cache[cache_key]
        else:
            client = provider.build_client(key, base_url=base, proxy=proxy)
            _client_cache[cache_key] = client
            while len(_client_cache) > _client_cache_max:
                _, old_client = _client_cache.popitem(last=False)
                try:
                    await old_client.close()
                except Exception:
                    pass

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
            saw_finish_reason = False
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

                # Usage 可能出现在两种位置：OpenAI 官方是最后一个 choices 为空的
                # 分块，而本网关实测是**和 finish_reason 挤在同一个分块里**（choices
                # 非空）。过去只在 choices 为空时找它，于是永远收不到 —— 落盘的 838
                # 条助手消息 usage 全是 {} 就是这么来的，连带真实 token、缓存命中
                # 率、压缩的 usage 锚点全部失效。两种布局都在这里收。
                usage_obj = getattr(chunk, "usage", None)
                if usage_obj:
                    partial.usage = {
                        "input": usage_obj.prompt_tokens or 0,
                        "output": usage_obj.completion_tokens or 0,
                        "total": usage_obj.total_tokens or 0,
                    }
                    details = getattr(usage_obj, "prompt_tokens_details", None)
                    cached = getattr(details, "cached_tokens", None) if details else None
                    if cached is not None:
                        # 前缀缓存命中量：把它留下，缓存行为才可观测。
                        partial.usage["cached"] = int(cached)

                # Skip chunks without choices
                if not chunk.choices:
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

                    # finish_reason 不等于流结束：OpenAI 兼容网关把 usage 放在**之后**
                    # 的最后一个分块里（本网关实测如此）。所以这里不再 yield+return，
                    # 而是继续读完流（上面 not chunk.choices 分支会收下 usage），
                    # 由下面统一的收尾发 done。否则 partial.usage 永远是 {}——落盘没有
                    # 真实 token、压缩的 usage 锚点失效、缓存命中率也无从观测。
                    saw_finish_reason = True
                    logger.debug("stream finish_reason=%s (draining for usage), chunks=%d",
                                 partial.stop_reason, chunk_count)
                    continue

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
            if saw_finish_reason:
                logger.debug("stream done: stop_reason=%s content_blocks=%d chunks=%d usage=%s",
                             partial.stop_reason, len(partial.content), chunk_count, partial.usage or "{}")
            else:
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
