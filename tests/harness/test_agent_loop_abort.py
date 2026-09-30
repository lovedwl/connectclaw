"""stream 异常结束的 abort 语义测试。

回归（2026-09-30 /stop 事故）：abort 杀掉的 LLM stream 会"无 done/error"
地结束（provider 契约是永不抛异常，取消被转成干净的生成器耗尽），旧逻辑
把它当正常完成——空白半截消息被当回复发送、还触发记忆抽取。现在 abort
信号置位时必须返回 stop_reason="aborted"，由 _run_loop 的 error/aborted
分支收尾（不落消息、不再续轮）。
"""

from __future__ import annotations

import asyncio

from connectclaw.agent import agent_loop
from connectclaw.agent.types import AgentLoopConfig
from connectclaw.provider.types import (
    AssistantMessage,
    Context,
    Model,
    StreamEvent,
    UserMessage,
)


def _partial(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[{"type": "text", "text": text}],
        model="test-model",
        timestamp=1.0,
    )


def _events_with_abort_midway(signal: asyncio.Event):
    async def _gen(*args, **kwargs):
        yield StreamEvent(type="start", partial=_partial(""))
        yield StreamEvent(type="text_delta", delta="喵", partial=_partial("喵"))
        signal.set()  # /stop 在流中途到达
        yield StreamEvent(type="text_delta", delta="～", partial=_partial("喵～"))
        # 没有 done/error —— abort 杀掉 stream 后的样子
    return _gen


async def _run(monkeypatch, stream_fn, signal):
    received = []

    async def emit(event):
        received.append(event)

    monkeypatch.setattr(agent_loop, "stream_simple", stream_fn)
    msg = await agent_loop._stream_assistant_response(
        Context(system_prompt="s", messages=[UserMessage(content="hi", timestamp=1.0)]),
        AgentLoopConfig(model=Model(id="test-model")),
        signal,
        emit,
    )
    return msg, received


def test_aborted_stream_returns_aborted_stop_reason(monkeypatch):
    signal = asyncio.Event()
    msg, received = asyncio.run(_run(monkeypatch, _events_with_abort_midway(signal), signal))
    assert msg.stop_reason == "aborted"
    # aborted 收尾不发 message_end —— 会话自动落盘只挂在这个事件上，
    # 所以半截消息不会进 session，也不会被当回复发出去
    assert all(e.get("type") != "message_end" for e in received)


def test_clean_stream_end_without_abort_still_degrades_to_stop(monkeypatch):
    # provider 真掉了流（非 abort）：维持旧行为，返回手头的 partial
    async def _gen_dropped(*args, **kwargs):
        yield StreamEvent(type="start", partial=_partial(""))
        yield StreamEvent(type="text_delta", delta="喵", partial=_partial("喵"))

    msg, _ = asyncio.run(_run(monkeypatch, _gen_dropped, asyncio.Event()))
    assert msg.stop_reason == "stop"
    assert msg.content[0]["text"] == "喵"
