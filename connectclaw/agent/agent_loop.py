"""
Agent loop — the heart of the system.

Double-loop pattern (mirrors pi-mono):
- OUTER LOOP: processes follow-up messages (queued after agent would stop)
- INNER LOOP: processes tool calls and steering messages
"""

import asyncio
import json
import time
from typing import Any, Awaitable, Callable

from connectclaw.logging import get_logger
from connectclaw.provider.stream import stream_simple
from connectclaw.provider.types import (
    AssistantMessage,
    Context,
    ToolDef,
    ToolResultMessage,
    UserMessage,
)

from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    ToolExecutionMode,
)

logger = get_logger(__name__)


AgentEventSink = Callable[[AgentEvent], Awaitable[None]]


# ── Public API ─────────────────────────────────────────────────


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None = None,
) -> list[AgentMessage]:
    """Start an agent loop with new prompt messages."""
    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )
    new_messages.extend(prompts)

    await emit({"type": "agent_start"})
    await emit({"type": "turn_start"})
    for prompt in prompts:
        await emit({"type": "message_start", "message": prompt})
        await emit({"type": "message_end", "message": prompt})

    await _run_loop(current_context, new_messages, config, signal, emit)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None = None,
) -> list[AgentMessage]:
    """Continue an agent loop from existing context."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    last = context.messages[-1]
    if last.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages],
        tools=context.tools,
    )

    await emit({"type": "agent_start"})
    await emit({"type": "turn_start"})

    await _run_loop(current_context, new_messages, config, signal, emit)
    return new_messages


# ── Main Loop ──────────────────────────────────────────────────


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> None:
    """Main loop logic shared by prompt and continue."""

    first_turn = True
    pending_messages: list[AgentMessage] = []
    # 2026-10 (F5 family): count of consecutive silent no-op nudges
    _noop_retries = 0

    if config.get_steering_messages:
        pending_messages = await config.get_steering_messages()

    # OUTER LOOP: follow-up messages
    while True:
        has_more_tool_calls = True

        # INNER LOOP: tool calls + steering
        while has_more_tool_calls or pending_messages:
            if signal and signal.is_set():
                await emit({"type": "agent_end", "messages": new_messages})
                return

            if not first_turn:
                await emit({"type": "turn_start"})
            else:
                first_turn = False

            # Inject pending messages before next assistant response
            if pending_messages:
                for msg in pending_messages:
                    await emit({"type": "message_start", "message": msg})
                    await emit({"type": "message_end", "message": msg})
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                pending_messages = []

            # Stream assistant response
            message = await _stream_assistant_response(
                current_context, config, signal, emit,
            )

            if message.stop_reason in ("error", "aborted"):
                await emit({
                    "type": "turn_end",
                    "message": message,
                    "tool_results": [],
                })
                await emit({"type": "agent_end", "messages": new_messages})
                return

            # 2026-10 (F5 family): the model sometimes ends a turn with thinking
            # only — no tool call, no visible text. The turn silently no-ops and
            # the user stares at a frozen progress card (2026-10-12: 24 min of
            # silence until the user poked us). Detect it and nudge the model to
            # continue, up to 2 times.
            if (
                message.stop_reason == "stop"
                and not any(c.get("type") == "toolCall" for c in message.content)
                and not any(
                    isinstance(b.get("text"), str) and b["text"].strip()
                    for b in message.content
                    if b.get("type") == "text"
                )
                and _noop_retries < 2
            ):
                _noop_retries += 1
                logger.warning(
                    "silent no-op turn (thinking only, no tool call, no text); nudging (%d/2)",
                    _noop_retries,
                )
                new_messages.append(message)
                await emit({
                    "type": "turn_end",
                    "message": message,
                    "tool_results": [],
                })
                # Injected at the top of the next inner-loop iteration
                pending_messages = [UserMessage(
                    content=(
                        "（系统提示）上一条回复没有产生任何可见内容或工具调用。"
                        "请继续完成上一步的操作：直接给出工具调用，或给出可见回复。"
                    ),
                    timestamp=time.time() * 1000,
                )]
                continue

            _noop_retries = 0
            new_messages.append(message)

            # Check for tool calls
            tool_calls = [c for c in message.content if c.get("type") == "toolCall"]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                # Notify live card of tool calls
                for tc in tool_calls:
                    if config.on_tool_call:
                        await config.on_tool_call(
                            tc.get("name", "?"),
                            tc.get("arguments", {}),
                            tc.get("id", ""),
                        )
                tool_results = await _execute_tool_calls(
                    current_context, message, tool_calls, config, signal, emit
                )
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)
                    # Notify live card of tool results
                    if config.on_tool_result:
                        res_text = ""
                        if hasattr(result, "content"):
                            for b in result.content:
                                if isinstance(b, dict) and b.get("type") == "text":
                                    res_text += b.get("text", "")
                        await config.on_tool_result(
                            result.tool_name if hasattr(result, "tool_name") else "?",
                            result.is_error if hasattr(result, "is_error") else False,
                            res_text,
                            result.tool_call_id if hasattr(result, "tool_call_id") else "",
                        )

            await emit({
                "type": "turn_end",
                "message": message,
                "tool_results": [tr.__dict__ if hasattr(tr, "__dict__") else tr for tr in tool_results],
            })

            # Check steering messages
            if config.get_steering_messages:
                pending_messages = await config.get_steering_messages()
            else:
                pending_messages = []

        # Check follow-up messages
        follow_up: list[AgentMessage] = []
        if config.get_follow_up_messages:
            follow_up = await config.get_follow_up_messages()

        if follow_up:
            pending_messages = follow_up
            continue

        break

    await emit({"type": "agent_end", "messages": new_messages})


# ── Assistant Response Streaming ───────────────────────────────


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> AssistantMessage:
    """Stream an assistant response from the LLM."""

    # Apply context transform if configured
    messages = list(context.messages)
    if config.transform_context:
        messages = await config.transform_context(messages, signal)

    # Convert to LLM-compatible messages
    llm_messages = messages
    if config.convert_to_llm:
        llm_messages = config.convert_to_llm(messages)
        # Filter to recognized roles (all messages are dataclasses now)
        llm_messages = [m for m in llm_messages if m.role in ("user", "assistant", "toolResult")]

        # Defensive: strip orphan tool_calls — assistant messages whose
        # tool_call_ids have no matching tool result messages.  Happens when
        # a session was interrupted mid-turn (tools issued but never executed).
        llm_messages = _strip_orphan_tool_calls(llm_messages)

    logger.debug("[STREAM] sending %d messages to LLM (after filter, from %d total)",
                 len(llm_messages), len(messages))

    # Build LLM context
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,  # type: ignore[arg-type]
        tools=[_tool_to_tooldef(t) for t in (context.tools or [])] if context.tools else None,
    )

    # Resolve API key
    api_key = config.api_key
    if not api_key and config.get_api_key:
        result = config.get_api_key(config.model.provider)
        if asyncio.iscoroutine(result):
            api_key = await result
        else:
            api_key = result

    partial_message: AssistantMessage | None = None
    added_partial = False
    event_count = 0
    thinking_start_time: float | None = None

    logger.debug("[STREAM] calling stream_simple(api_key=%s, base_url=%s, reasoning=%s)",
                 "***" if api_key else "None", config.model.base_url, config.reasoning)

    async for event in stream_simple(
        config.model,
        llm_context,
        api_key=api_key,
        signal=signal,
        reasoning=config.reasoning,
        session_id=config.session_id,
        max_retries=config.max_retries,
        base_url=config.model.base_url,
        proxy=config.model.proxy,
    ):
        event_count += 1
        if signal and signal.is_set():
            break

        # Debug first few events
        if event_count <= 5:
            logger.debug("[STREAM] event[%d]: %s delta_len=%d error=%s",
                         event_count, event.type, len(event.delta or ""), event.error_message)

        match event.type:
            case "start":
                if event.partial:
                    partial_message = event.partial
                    context.messages.append(partial_message)  # type: ignore[arg-type]
                    added_partial = True
                    await emit({
                        "type": "message_start",
                        "message": _message_to_dict(partial_message),
                    })

            case "thinking_delta":
                if event.partial and partial_message:
                    partial_message = event.partial
                    context.messages[-1] = partial_message  # type: ignore[index]
                    if thinking_start_time is None:
                        thinking_start_time = time.time()
                    if config.on_thinking_delta and event.delta:
                        await config.on_thinking_delta(event.delta)
                    await emit({
                        "type": "message_update",
                        "message": _message_to_dict(partial_message),
                        "stream_event": event.__dict__,
                    })

            case "text_delta":
                if event.partial and partial_message:
                    partial_message = event.partial
                    context.messages[-1] = partial_message  # type: ignore[index]
                    # End thinking phase if it was active
                    if thinking_start_time is not None:
                        if config.on_thinking_done:
                            await config.on_thinking_done(time.time() - thinking_start_time)
                        thinking_start_time = None
                    if config.on_text_delta and event.delta:
                        await config.on_text_delta(event.delta)
                    await emit({
                        "type": "message_update",
                        "message": _message_to_dict(partial_message),
                        "stream_event": event.__dict__,
                    })

            case "toolcall_delta":
                if event.partial and partial_message:
                    partial_message = event.partial
                    context.messages[-1] = partial_message  # type: ignore[index]
                    await emit({
                        "type": "message_update",
                        "message": _message_to_dict(partial_message),
                        "stream_event": event.__dict__,
                    })

            case "done":
                if event.message:
                    final = event.message
                    logger.debug("[STREAM] done: stop_reason=%s content_blocks=%d",
                                 final.stop_reason, len(final.content))
                    if final.stop_reason == "error":
                        logger.debug("[STREAM]   error_message=%s", final.error_message)
                    # End thinking phase if active
                    if thinking_start_time is not None:
                        if config.on_thinking_done:
                            await config.on_thinking_done(time.time() - thinking_start_time)
                        thinking_start_time = None
                    if config.on_text_done:
                        await config.on_text_done()
                    if added_partial:
                        context.messages[-1] = final  # type: ignore[index]
                    else:
                        context.messages.append(final)  # type: ignore[arg-type]
                        await emit({
                            "type": "message_start",
                            "message": _message_to_dict(final),
                        })
                    await emit({
                        "type": "message_end",
                        "message": _message_to_dict(final),
                    })
                    return final

            case "error":
                logger.error("[STREAM] ERROR: %s", event.error_message)
                if thinking_start_time is not None and config.on_thinking_done:
                    await config.on_thinking_done(time.time() - thinking_start_time)
                thinking_start_time = None
                if config.on_text_done:
                    await config.on_text_done()
                error_msg = AssistantMessage(
                    content=[{"type": "text", "text": ""}],
                    model=config.model.id,
                    stop_reason="error",
                    usage={},
                    error_message=event.error_message,
                    timestamp=time.time() * 1000,
                )
                if added_partial:
                    context.messages[-1] = error_msg  # type: ignore[index]
                await emit({
                    "type": "message_end",
                    "message": _message_to_dict(error_msg),
                })
                return error_msg

    logger.warning("[STREAM] stream ended with %d events, no done/error", event_count)
    # abort 杀掉的 stream（连接被关、生成器正常耗尽）会走到这里。若不区分，
    # "已中断"会被当成"正常完成"：半截空白消息落进会话、被发到聊天、
    # 还触发 confirm_usage / 记忆抽取。检查 abort 信号，按中断收尾。
    if signal is not None and signal.is_set():
        return AssistantMessage(
            content=[{"type": "text", "text": ""}],
            model=config.model.id,
            stop_reason="aborted",
            usage={},
            timestamp=time.time() * 1000,
        )
    # If stream ended without done/error, return whatever we have
    final = partial_message or AssistantMessage(
        content=[{"type": "text", "text": ""}],
        model=config.model.id,
        stop_reason="stop",
        usage={},
        timestamp=time.time() * 1000,
    )
    return final


# ── Tool Execution ─────────────────────────────────────────────


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[dict[str, Any]],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    """Execute tool calls. Supports sequential and parallel modes."""

    if config.tool_execution == "sequential":
        return await _execute_sequential(
            current_context, assistant_message, tool_calls, config, signal, emit
        )
    return await _execute_parallel(
        current_context, assistant_message, tool_calls, config, signal, emit
    )


async def _execute_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[dict[str, Any]],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    results: list[ToolResultMessage] = []

    for tc in tool_calls:
        if signal and signal.is_set():
            break
        try:
            result_msg = await _execute_single_tool(
                current_context, assistant_message, tc, config, signal, emit
            )
        except Exception as e:
            err_result = AgentToolResult(
                content=[{"type": "text", "text": str(e)}],
                details={"is_error": True},
            )
            result_msg = ToolResultMessage(
                tool_call_id=tc["id"],
                tool_name=tc["name"],
                content=err_result.content,
                is_error=True,
                timestamp=time.time() * 1000,
            )
            # M4：预处理失败也要发 tool_execution_end，否则调用方
            # pending_tool_calls 里这条永久残留（与并行路径行为一致）。
            await emit({
                "type": "tool_execution_end",
                "tool_call_id": tc["id"],
                "tool_name": tc["name"],
                "result": err_result.__dict__,
                "is_error": True,
            })
        results.append(result_msg)
        await emit({"type": "message_start", "message": _message_to_dict(result_msg)})
        await emit({"type": "message_end", "message": _message_to_dict(result_msg)})

    return results


async def _execute_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[dict[str, Any]],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    """Parallel execution: preflight sequentially, execute concurrently."""

    # Phase 1: Preflight — find tools, validate args, fire auth hooks.
    # ALL tools start preflight concurrently so that every auth card is
    # sent at once.  asyncio.gather waits for all of them before we
    # proceed to Phase 2 (execution).
    preparations: list[tuple[dict, AgentTool, dict]] = []
    preflight_errors: dict[str, AgentToolResult] = {}  # tool_call_id → error

    for tc in tool_calls:
        await emit({
            "type": "tool_execution_start",
            "tool_call_id": tc["id"],
            "tool_name": tc["name"],
            "args": tc.get("arguments", {}),
        })

    logger.debug("[PREFLIGHT] launching %d tools concurrently", len(tool_calls))

    async def _preflight_one(tc: dict) -> tuple[dict, AgentTool, dict] | None:
        t0 = time.time()
        try:
            tool, validated_args = await _prepare_tool_call(
                current_context, tc, config, signal
            )
            logger.debug("[PREFLIGHT] %s(%s) ok in %.1fs",
                         tc["name"], tc["id"][:8], time.time() - t0)
            return (tc, tool, validated_args)
        except Exception as e:
            logger.debug("[PREFLIGHT] %s(%s) failed in %.1fs: %s",
                         tc["name"], tc["id"][:8], time.time() - t0, e)
            preflight_errors[tc["id"]] = AgentToolResult(
                content=[{"type": "text", "text": str(e)}],
                details=None,
            )
            return None

    preflight_tasks = [_preflight_one(tc) for tc in tool_calls]
    preflight_results = await asyncio.gather(*preflight_tasks)
    logger.debug("[PREFLIGHT] all done — %d ok, %d errors",
                 len([r for r in preflight_results if r is not None]),
                 len(preflight_errors))
    for result in preflight_results:
        if result is not None:
            tc, tool, validated_args = result
            preparations.append((tc, tool, validated_args))

    # Phase 2: Execute prepared tools concurrently
    # M3：/stop 前置检查——顺序路径有，并行路径此前没有，abort 后
    # prepared 的工具仍会全量执行。
    aborted_before_exec = bool(signal and signal.is_set())

    async def run_one(tc: dict, tool: AgentTool, args: dict) -> AgentToolResult:
        try:
            result = await tool.execute(
                tc["id"], args, signal=signal,
                on_update=_make_tool_update(config, tc),
            )
            return result
        except Exception as e:
            return AgentToolResult(
                content=[{"type": "text", "text": str(e)}],
                details=None,
            )

    if aborted_before_exec:
        # 已 abort：不执行，但为每个 prepared 调用留一个占位，Phase 3
        # 会补发中止结果保持 tool_call/result 配对。
        executed: list[Any] = [None] * len(preparations)
    else:
        tasks = [run_one(tc, tool, args) for tc, tool, args in preparations]
        executed = await asyncio.gather(*tasks, return_exceptions=True)

    # Phase 3: Finalize in source order
    # M3：这里不再提前 break——工具已经执行完，break 会丢掉真实结果，
    # 制造孤儿 toolCall；abort 时统一补中止结果。
    results: list[ToolResultMessage] = []
    for (tc, tool, args), exec_result in zip(preparations, executed):
        if exec_result is None:
            result = AgentToolResult(
                content=[{"type": "text", "text": "已中止（/stop）：工具未执行"}],
                details={"is_error": True},
            )
            is_error = True
        elif isinstance(exec_result, Exception):
            result = AgentToolResult(
                content=[{"type": "text", "text": str(exec_result)}],
                details=None,
            )
            is_error = True
        elif isinstance(exec_result, AgentToolResult):
            result = exec_result
            is_error = _result_is_error(exec_result)
        else:
            result = AgentToolResult(
                content=[{"type": "text", "text": f"Unexpected result: {exec_result}"}],
                details=None,
            )
            is_error = True

        # Call after_tool_call hook
        if config.after_tool_call:
            hook_result = await config.after_tool_call(
                {
                    "tool_call": tc,
                    "result": result,
                    "is_error": is_error,
                    "context": current_context,
                },
                signal,
            )
            if hook_result and hook_result.get("result"):
                result = hook_result["result"]
            if hook_result and "is_error" in hook_result:
                is_error = hook_result["is_error"]

        await emit({
            "type": "tool_execution_end",
            "tool_call_id": tc["id"],
            "tool_name": tc["name"],
            "result": result.__dict__,
            "is_error": is_error,
        })

        msg = ToolResultMessage(
            tool_call_id=tc["id"],
            tool_name=tc["name"],
            content=result.content,
            is_error=is_error,
            timestamp=time.time() * 1000,
        )
        await emit({"type": "message_start", "message": _message_to_dict(msg)})
        await emit({"type": "message_end", "message": _message_to_dict(msg)})
        results.append(msg)

    # Emit results for tools that failed during preflight (auth denied etc.)
    for tc_id, err_result in preflight_errors.items():
        # Find the original tool call dict
        tc = next((t for t in tool_calls if t["id"] == tc_id), None)
        if tc is None:
            continue
        await emit({
            "type": "tool_execution_end",
            "tool_call_id": tc_id,
            "tool_name": tc["name"],
            "result": err_result.__dict__,
            "is_error": True,
        })
        msg = ToolResultMessage(
            tool_call_id=tc_id,
            tool_name=tc["name"],
            content=err_result.content,
            is_error=True,
            timestamp=time.time() * 1000,
        )
        await emit({"type": "message_start", "message": _message_to_dict(msg)})
        await emit({"type": "message_end", "message": _message_to_dict(msg)})
        results.append(msg)

    return results


async def _execute_single_tool(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tc: dict[str, Any],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> ToolResultMessage:
    """Execute a single tool call (sequential mode)."""

    await emit({
        "type": "tool_execution_start",
        "tool_call_id": tc["id"],
        "tool_name": tc["name"],
        "args": tc.get("arguments", {}),
    })

    tool, validated_args = await _prepare_tool_call(
        current_context, tc, config, signal
    )
    result, is_error = await _execute_prepared_tool(tc, tool, validated_args, signal, config)

    if config.after_tool_call:
        hook_result = await config.after_tool_call(
            {
                "tool_call": tc,
                "result": result,
                "is_error": is_error,
                "context": current_context,
            },
            signal,
        )
        if hook_result and hook_result.get("result"):
            result = hook_result["result"]
        if hook_result and "is_error" in hook_result:
            is_error = hook_result["is_error"]

    await emit({
        "type": "tool_execution_end",
        "tool_call_id": tc["id"],
        "tool_name": tc["name"],
        "result": result.__dict__,
        "is_error": is_error,
    })

    msg = ToolResultMessage(
        tool_call_id=tc["id"],
        tool_name=tc["name"],
        content=result.content,
        is_error=is_error,
        timestamp=time.time() * 1000,
    )
    return msg


async def _prepare_tool_call(
    current_context: AgentContext,
    tc: dict[str, Any],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> tuple[AgentTool, dict[str, Any]]:
    """Find tool, validate args, call before_tool_call hook."""
    tool = None
    for t in (current_context.tools or []):
        if t.name == tc["name"]:
            tool = t
            break

    if tool is None:
        raise RuntimeError(f"Tool '{tc['name']}' not found")

    args = tc.get("arguments", {})

    # 2026-10 (F5): the stream parser marks truncated tool-call args with
    # args_error — refuse to run the tool with empty params, surface a clear
    # error so the model retries or splits the call.
    if tc.get("args_error"):
        raise RuntimeError(tc["args_error"])
    # Call before_tool_call hook
    if config.before_tool_call:
        hook_result = await config.before_tool_call(
            {
                "tool_call": tc,
                "args": args,
                "context": current_context,
            },
            signal,
        )
        if hook_result and hook_result.get("block"):
            raise RuntimeError(hook_result.get("reason", "Tool execution blocked"))

    return tool, args


def _make_tool_update(config: AgentLoopConfig, tc: dict[str, Any]):
    """Build an on_update callback forwarding mid-execution tool updates to the
    live card via config.on_tool_update. Returns None when no sink is set."""
    if not config.on_tool_update:
        return None

    async def _on_update(update: AgentToolResult) -> None:
        await config.on_tool_update(tc["id"], tc["name"], update)

    return _on_update


def _result_is_error(result: AgentToolResult) -> bool:
    """Honor a tool's self-reported logical failure (details['is_error']).
    
    2026-10 fix: tool results used to be flattened to is_error=False in the
    loop, silencing refusals/validation failures (hash_edit E_BAD_REF,
    protected files, ...) so the harness/Feishu card never saw them as errors.
    """
    details = getattr(result, "details", None)
    return bool(isinstance(details, dict) and details.get("is_error"))

async def _execute_prepared_tool(
    tc: dict[str, Any],
    tool: AgentTool,
    args: dict[str, Any],
    signal: asyncio.Event | None,
    config: AgentLoopConfig | None = None,
) -> tuple[AgentToolResult, bool]:
    """Execute a prepared tool call."""
    try:
        on_update = _make_tool_update(config, tc) if config else None
        result = await tool.execute(tc["id"], args, signal=signal, on_update=on_update)
        return result, _result_is_error(result)
    except Exception as e:
        return (
            AgentToolResult(
                content=[{"type": "text", "text": str(e)}],
                details=None,
            ),
            True,
        )


# ── Helpers ────────────────────────────────────────────────────


def _strip_orphan_tool_calls(messages: list) -> list:
    """Remove tool_calls from assistant messages that lack matching tool results.

    If a session was interrupted mid-turn (e.g. the agent issued tool calls but
    crashed before executing them), the stored history contains an assistant
    message with tool_calls but no ToolResultMessage for those call IDs.
    The API rejects such messages with a 400 error.
    """
    # Collect all tool_call_ids that have matching tool result messages.
    satisfied_ids: set[str] = set()
    for m in messages:
        if m.role == "toolResult" and hasattr(m, "tool_call_id") and m.tool_call_id:
            satisfied_ids.add(m.tool_call_id)

    result = []
    for m in messages:
        if m.role != "assistant":
            result.append(m)
            continue

        content = getattr(m, "content", [])
        if not isinstance(content, list):
            result.append(m)
            continue

        # Split content into tool_calls and non-tool-call blocks.
        tool_blocks = [b for b in content if b.get("type") == "toolCall"]
        non_tool_blocks = [b for b in content if b.get("type") != "toolCall"]

        if not tool_blocks:
            result.append(m)
            continue

        # Check which tool_calls in this message have matching results.
        # M5：只剥孤儿 tool_call——此前只要有一个无结果就把整条消息的
        # 全部 tool_calls 剥掉，同消息里"有结果"的调用被剥后，其
        # toolResult 反过来变成孤儿（API 400）。
        orphan_ids = {
            tb.get("id", "") for tb in tool_blocks
            if tb.get("id", "") and tb.get("id") not in satisfied_ids
        }

        if not orphan_ids:
            result.append(m)
            continue

        kept_tool_blocks = [
            tb for tb in tool_blocks if tb.get("id", "") not in orphan_ids
        ]

        # 还有保留的 toolCall：只替换掉孤儿块，消息结构不变。
        if kept_tool_blocks:
            get_logger(__name__).warning(
                "Stripping %d orphan tool_call(s) from assistant message "
                "(kept %d with results — session was interrupted mid-turn)",
                len(orphan_ids), len(kept_tool_blocks),
            )
            stripped = type(m)(
                role="assistant",
                content=list(non_tool_blocks) + kept_tool_blocks,
                model=getattr(m, "model", ""),
                stop_reason=getattr(m, "stop_reason", "stop"),
                usage=getattr(m, "usage", {}),
                error_message=getattr(m, "error_message", None),
                timestamp=getattr(m, "timestamp", 0.0),
                duration_ms=getattr(m, "duration_ms", 0.0),
                ttft_ms=getattr(m, "ttft_ms", 0.0),
                retry_count=getattr(m, "retry_count", 0),
            )
            result.append(stripped)
            continue

        # Orphan found — strip tool_calls, keep text + thinking only.
        # If nothing is left after stripping (all blocks were orphan
        # tool calls), drop the message entirely to avoid a 400 error
        # from the API ("content or tool_calls must be set").
        meaningful = [b for b in non_tool_blocks
                      if b.get("type") in ("text", "thinking")]
        if not meaningful:
            get_logger(__name__).warning(
                "Dropping orphan assistant message (%d tool call(s), "
                "no text/thinking left — session was interrupted mid-turn)",
                len(tool_blocks),
            )
            continue

        get_logger(__name__).warning(
            "Stripping orphan tool_calls from assistant message (%d tool call(s), "
            "missing tool results — session was interrupted mid-turn)",
            len(tool_blocks),
        )

        # Some LLM APIs (e.g. DeepSeek) require assistant.content to be
        # a non-empty string — thinking-only blocks are NOT accepted as
        # content.  If stripping left us with only thinking blocks, promote
        # the last thinking block's text into a text block so the API doesn't
        # reject the message with 400.
        if not any(b.get("type") == "text" for b in non_tool_blocks):
            thinking_texts = [
                b.get("thinking", "") for b in non_tool_blocks
                if b.get("type") == "thinking"
            ]
            if thinking_texts:
                non_tool_blocks = list(non_tool_blocks) + [
                    {"type": "text", "text": thinking_texts[-1]}
                ]

        stripped = type(m)(
            role="assistant",
            content=non_tool_blocks,
            model=getattr(m, "model", ""),
            stop_reason=getattr(m, "stop_reason", "stop"),
            usage=getattr(m, "usage", {}),
            error_message=getattr(m, "error_message", None),
            timestamp=getattr(m, "timestamp", 0.0),
            duration_ms=getattr(m, "duration_ms", 0.0),
            ttft_ms=getattr(m, "ttft_ms", 0.0),
            retry_count=getattr(m, "retry_count", 0),
        )
        result.append(stripped)

    return result


def _tool_to_tooldef(tool: AgentTool):
    """Convert AgentTool to ToolDef for the provider layer."""
    return ToolDef(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


def _message_to_dict(msg) -> dict[str, Any]:
    """Return message as-is — keep dataclass intact so state.messages uses .role."""
    return msg
