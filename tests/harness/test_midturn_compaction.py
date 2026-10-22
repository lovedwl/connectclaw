"""回归（2026-10-21）：压缩只在「用户发消息」时才检查，turn 内部永不压缩。

事故形态
--------
主人观察到：上下文超过阈值后并不会立刻自动压缩，而是**等他发下一条消息**
才压。根因是 ``should_compact()`` 全代码库只有一个调用点 ——
``agent_harness._execute_turn()`` 开头，而 ``_execute_turn`` 每收到一条用户
消息只跑一次。真正干活的内层循环 ``agent_loop._run_loop()`` 里，一个 turn
可以跑几十次工具调用、往上下文里灌几 MB 工具输出，**全程没有任何 token
检查**。

下游后果比"晚一点压缩"严重：单轮增长顶穿 context window 时，下一次 LLM
请求直接 400（context_length_exceeded），那一轮当场崩。

契约
----
``AgentLoopConfig.compact_if_needed(context)`` 在**每次发起 LLM 请求前**
被调用（内层循环每一轮一次），回调可原地替换 ``context.messages``，从而让
本轮后续请求立即瘦身。未配置时行为完全不变。
"""

from __future__ import annotations

import asyncio

from connectclaw.agent import agent_loop
from connectclaw.agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
)
from connectclaw.provider.types import (
    AssistantMessage,
    Model,
    StreamEvent,
    UserMessage,
)


class _NoopTool(AgentTool):
    name = "noop"
    label = "noop"
    description = "noop"
    parameters: dict = {}
    execution_mode = "parallel"

    async def execute(self, tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[{"type": "text", "text": "ok"}])


def _scripted_llm(n_tool_turns):
    """Fake stream_simple: first n turns call noop, then answer with text."""
    state = {"i": 0}
    seen_contexts = []

    def _factory(model, ctx, **kwargs):
        async def _gen():
            turn = state["i"]
            state["i"] += 1
            seen_contexts.append(list(ctx.messages))
            if turn < n_tool_turns:
                msg = AssistantMessage(
                    content=[{
                        "type": "toolCall",
                        "id": "call_%d" % turn,
                        "name": "noop",
                        "arguments": {},
                    }],
                    model="test-model",
                    stop_reason="toolUse",
                    timestamp=1.0,
                )
            else:
                msg = AssistantMessage(
                    content=[{"type": "text", "text": "done"}],
                    model="test-model",
                    stop_reason="stop",
                    timestamp=1.0,
                )
            yield StreamEvent(type="start", partial=msg)
            yield StreamEvent(type="done", message=msg)
        return _gen()

    return _factory, seen_contexts


def _run_loop(monkeypatch, config, ctx, n_tool_turns):
    factory, seen = _scripted_llm(n_tool_turns)
    monkeypatch.setattr(agent_loop, "stream_simple", factory)
    events = []

    async def emit(event):
        events.append(event)

    ctx.messages.append(UserMessage(content="跑个长任务", timestamp=1.0))
    asyncio.run(agent_loop._run_loop(ctx, [], config, None, emit))
    return seen, events


def test_compact_checked_before_every_llm_call(monkeypatch):
    """核心回归：3 次 LLM 请求 → 压缩闸被调用 3 次（不是 0 次）。"""
    calls = []

    async def compact_if_needed(context):
        calls.append(len(context.messages))

    config = AgentLoopConfig(
        model=Model(id="test-model"),
        compact_if_needed=compact_if_needed,
    )
    ctx = AgentContext(system_prompt="s", messages=[], tools=[_NoopTool()])
    _run_loop(monkeypatch, config, ctx, n_tool_turns=2)

    assert calls == [1, 3, 5], "expected 3 checks, got %r" % (calls,)


def test_compact_callback_can_shrink_context_midturn(monkeypatch):
    """回调替换 current_context.messages 后，本轮后续请求立刻变短。"""
    seen_lengths = []

    async def compact_if_needed(context):
        seen_lengths.append(len(context.messages))
        if len(context.messages) > 2:
            context.messages = context.messages[-1:]

    config = AgentLoopConfig(
        model=Model(id="test-model"),
        compact_if_needed=compact_if_needed,
    )
    ctx = AgentContext(system_prompt="s", messages=[], tools=[_NoopTool()])
    seen, _ = _run_loop(monkeypatch, config, ctx, n_tool_turns=2)

    assert len(seen[2]) == 1, "mid-turn compaction did not take effect: %d" % len(seen[2])
    # 回调每次都在请求前跑：1(仅 user) → 3(压缩前) → 3(压缩后只留 1 条 + 本轮 2 条)
    assert seen_lengths == [1, 3, 3]


def test_no_callback_keeps_old_behavior(monkeypatch):
    """没配回调时一切照旧（不崩、不改变消息数量）。"""
    config = AgentLoopConfig(model=Model(id="test-model"))
    ctx = AgentContext(system_prompt="s", messages=[], tools=[_NoopTool()])
    seen, events = _run_loop(monkeypatch, config, ctx, n_tool_turns=1)

    assert len(seen) == 2
    assert len(seen[1]) == 3
    assert any(e.get("type") == "agent_end" for e in events)


def test_compact_failure_does_not_kill_turn(monkeypatch):
    """压缩回调抛异常不能掀翻整个 turn —— 降级为不压缩，继续跑。"""
    async def boom(context):
        raise RuntimeError("summarizer exploded")

    config = AgentLoopConfig(
        model=Model(id="test-model"),
        compact_if_needed=boom,
    )
    ctx = AgentContext(system_prompt="s", messages=[], tools=[_NoopTool()])
    seen, events = _run_loop(monkeypatch, config, ctx, n_tool_turns=1)

    assert len(seen) == 2
    assert any(e.get("type") == "agent_end" for e in events)


# ── harness 接线测试 ───────────────────────────────────────────
# 上面 4 条覆盖 agent_loop 的契约；这里守住「harness 真的把闸接上了」——
# 接线漏了是无声失效（调用点为空，测试不红，线上照旧只在用户发消息时压缩）。

def _bare_harness(monkeypatch, *, compacted: bool, replace_with=None):
    from connectclaw.agent.harness import agent_harness as ah
    from connectclaw.agent.harness.compaction import CompactionSettings
    from connectclaw.provider.types import Model

    h = object.__new__(ah.AgentHarness)
    h._model = Model(id="test-model", context_window=1000)
    h._compaction_settings = CompactionSettings(
        enabled=True, reserve_tokens=100, keep_recent_tokens=200
    )
    h._persist_tasks = set()
    h._get_api_key = None
    h._midturn_compacting = False
    h._skip_gate_once = False

    class _Sess:
        async def get_path_to_root(self):
            return []

    h._session = _Sess()

    async def _fake_run_compaction(api_key, messages, tokens):
        return compacted

    monkeypatch.setattr(h, "_run_compaction", _fake_run_compaction)
    monkeypatch.setattr(
        ah, "build_session_context",
        lambda entries: _NS(messages=list(replace_with or [])),
    )
    return h


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_midturn_gate_skips_when_under_threshold(monkeypatch):
    """没超阈值就不该碰 session / 不该压缩。"""
    import asyncio as _a

    h = _bare_harness(monkeypatch, compacted=True, replace_with=["X"])
    ctx = _NS(messages=[UserMessage(content="短", timestamp=1.0)])
    _a.run(h._compact_mid_turn(ctx))
    assert len(ctx.messages) == 1  # 未被替换


def test_midturn_gate_replaces_messages_when_over_threshold(monkeypatch):
    """超阈值 + 压缩成功 → context.messages 被压缩后的上下文替换。"""
    import asyncio as _a

    h = _bare_harness(monkeypatch, compacted=True, replace_with=["摘要", "最近"])
    big = [UserMessage(content="喵" * 5000, timestamp=1.0) for _ in range(4)]
    ctx = _NS(messages=list(big))
    _a.run(h._compact_mid_turn(ctx))
    assert ctx.messages == ["摘要", "最近"]


def test_midturn_gate_no_replace_when_compaction_declined(monkeypatch):
    """benefit guard 拒绝压缩（返回 False）时不动上下文。"""
    import asyncio as _a

    h = _bare_harness(monkeypatch, compacted=False, replace_with=["不该出现"])
    big = [UserMessage(content="喵" * 5000, timestamp=1.0) for _ in range(4)]
    ctx = _NS(messages=list(big))
    _a.run(h._compact_mid_turn(ctx))
    assert len(ctx.messages) == 4
    assert h._midturn_compacting is False  # 锁必须释放


def test_midturn_gate_reentrancy_guard(monkeypatch):
    """压缩中再进来直接返回，不会套娃发 LLM 请求。"""
    import asyncio as _a

    h = _bare_harness(monkeypatch, compacted=True, replace_with=["X"])
    h._midturn_compacting = True
    big = [UserMessage(content="喵" * 5000, timestamp=1.0) for _ in range(4)]
    ctx = _NS(messages=list(big))
    _a.run(h._compact_mid_turn(ctx))
    assert len(ctx.messages) == 4


def test_first_request_of_turn_is_not_blocked_by_compaction(monkeypatch):
    """2026-10：turn 开头判断"不紧急"→ 首请求跳过压缩闸（先让模型开口），
    后续请求才恢复自动压缩。此前首请求前就压，用户只看到"思考中"干等几分钟。"""
    import asyncio as _a

    h = _bare_harness(monkeypatch, compacted=True, replace_with=["压缩后"])
    h._skip_gate_once = True
    big = [UserMessage(content="喵" * 5000, timestamp=1.0) for _ in range(4)]

    ctx = _NS(messages=list(big))
    _a.run(h._compact_mid_turn(ctx))
    assert len(ctx.messages) == 4, "首请求不该被压缩拦下"
    assert h._skip_gate_once is False, "跳过标记必须只用一次"

    # 下一次请求（工具调用之后）恢复正常压缩
    _a.run(h._compact_mid_turn(ctx))
    assert ctx.messages == ["压缩后"]
