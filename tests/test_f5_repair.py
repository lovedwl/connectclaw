"""F5 根治回归测试（2026-10）。

F5：大体积 hash_edit payload 的 toolCall 参数被静默弄坏。三个根因修复：

1. hashline/parse.py —— 模型把 edits 数组双编码成 JSON 字符串且丢掉
   "op" 前的 ']'（键落进数组里 → json.loads 失败 → "edits 为空"）。
   ``_repair_edits_json_string`` 用结构状态机把数组修回来。
2. provider/stream.py —— 截断的 arguments JSON 不再静默变 ``{}``，
   toolCall 槽位打上 ``args_error`` 标记（+ 警告日志）。
3. agent/agent_loop.py —— 带 ``args_error`` 的调用被拒执行（不再拿空
   参数跑工具）；静默 no-op turn（只有 thinking）自动 nudge ≤2 次，
   不再卡 24 分钟等主人戳。
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from connectclaw.hashline.parse import _coerce_edits_array, _repair_edits_json_string

# 真实事故 payload（session cba6615054b4，2026-10-12 06:16 UTC，
# 对 model_card.py 的 hash_edit）：双编码字符串，"op" 前的 ']' 被丢。
_INCIDENT_EDITS = '[{"lines": ["    elements.append({\\"tag\\": \\"hr\\"})", "    elements.append(_row(_button(\\n        \\"🔍 自动发现新模型（查列表+上下文）\\",\\n        {\\"kind\\": KIND, \\"stage\\": \\"discover\\", \\"provider\\": provider},\\n        \\"primary\\", nonce=nonce,\\n    )))", "op": "replace", "pos": "142#XY"}]'


def test_repair_recovers_real_incident_payload():
    arr = _repair_edits_json_string(_INCIDENT_EDITS)
    assert isinstance(arr, list) and len(arr) == 1
    item = arr[0]
    assert item["op"] == "replace"
    assert item["pos"] == "142#XY"
    assert len(item["lines"]) == 2
    assert item["lines"][0] == "    elements.append({\"tag\": \"hr\"})"


def test_repair_synthetic_missing_bracket():
    mangled = (
        '[{"lines": ["B","op": "replace", "pos": "2#AB"}, '
        '{"op": "replace_text", "oldText": "c", "newText": "C"}]'
    )
    arr = _repair_edits_json_string(mangled)
    assert [i["op"] for i in arr] == ["replace", "replace_text"]
    assert arr[0]["lines"] == ["B"]
    assert arr[0]["pos"] == "2#AB"
    assert arr[1]["oldText"] == "c"


def test_repair_truncated_returns_none():
    # 字符串中途截断：内容已丢，修不回来
    assert _repair_edits_json_string('[{"lines": ["ab') is None
    # 结尾多余括号补上后仍不是合法对象 → 拒绝硬修
    assert _repair_edits_json_string('[{"op": "replace", "pos": "1#AB")') is None


def test_coerce_json_then_repair_then_literal():
    clean = json.dumps([{"op": "replace_text", "oldText": "a", "newText": "b"}])
    assert _coerce_edits_array(clean) == [
        {"op": "replace_text", "oldText": "a", "newText": "b"}
    ]
    mangled = '[{"lines": ["x","op": "replace", "pos": "1#AB"}]'
    assert _coerce_edits_array(mangled) == [
        {"lines": ["x"], "op": "replace", "pos": "1#AB"}
    ]
    # 非 list 的解析结果原样透传，交给下游精确报错
    assert _coerce_edits_array('{"a": 1}') == '{"a": 1}'
    # Python 字面量兜底
    assert _coerce_edits_array("['a', 'b']") == ["a", "b"]
    # 非字符串原样返回
    assert _coerce_edits_array([1, 2]) == [1, 2]


# ── hash_edit 端到端：事故形态的 edits 字符串应被修复并执行 ──


def test_hash_edit_executes_mangled_edits(tmp_path):
    from connectclaw.coding.tools.hash_edit import create_hash_edit_tool
    from connectclaw.hashline.hash import compute_line_hash

    f = tmp_path / "t.py"
    f.write_text("a\nb\nc\n")
    tool = create_hash_edit_tool(str(tmp_path))
    anchor = compute_line_hash(["a", "b", "c"], 1)  # 第 2 行（0-based）
    clean = [
        {"lines": ["B"], "op": "replace", "pos": f"2#{anchor}"},
        {"op": "replace_text", "oldText": "c", "newText": "C"},
    ]
    s = json.dumps(clean, ensure_ascii=False)
    # 复刻 F5：模型把 "lines" 数组的 ']' 丢了
    s = s.replace('"], "op"', '", "op"', 1)
    assert s != json.dumps(clean, ensure_ascii=False)
    res = asyncio.run(tool.execute("call_1", {"path": str(f), "edits": s}))
    assert not (res.details or {}).get("is_error"), res.content
    assert f.read_text() == "a\nB\nC\n"


# ── stream 解析器：截断参数 → args_error 标记 ─────────────────


def _chunk(delta, finish=None):
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
    )


def test_stream_truncated_toolcall_args_marked(monkeypatch):
    from connectclaw.provider import stream as stream_mod
    from connectclaw.provider.types import Context, Model, UserMessage

    tc1 = SimpleNamespace(
        index=0, id="call_1",
        function=SimpleNamespace(name="hash_edit", arguments='{"path": "/tmp/x"')
    )
    tc2 = SimpleNamespace(
        index=0, id=None,
        function=SimpleNamespace(name=None, arguments=', "edits": [{"op": "repl')
    )
    chunks = [
        _chunk(SimpleNamespace(content=None, tool_calls=[tc1])),
        _chunk(SimpleNamespace(content=None, tool_calls=[tc2])),
        _chunk(SimpleNamespace(content=None, tool_calls=None), finish="tool_calls"),
    ]

    async def create(**kwargs):
        async def _gen():
            for c in chunks:
                yield c
        return _gen()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(
        stream_mod._provider, "build_client",
        lambda key, base_url=None, proxy=None: fake_client,
    )

    async def main():
        evs = []
        async for ev in stream_mod.stream_simple(
            Model(id="m", base_url="http://x"),
            Context(system_prompt="s",
                    messages=[UserMessage(content="hi", timestamp=1.0)]),
        ):
            evs.append(ev)
        return evs

    evs = asyncio.run(main())
    done = [e for e in evs if e.type == "done"]
    assert len(done) == 1
    msg = done[0].message
    assert msg.stop_reason == "toolUse"
    tcs = [c for c in msg.content if c.get("type") == "toolCall"]
    assert len(tcs) == 1
    assert tcs[0]["arguments"] == {}
    assert "截断" in tcs[0].get("args_error", "")


def test_stream_complete_toolcall_args_still_fine(monkeypatch):
    from connectclaw.provider import stream as stream_mod
    from connectclaw.provider.types import Context, Model, UserMessage

    chunks = [
        _chunk(SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                index=0, id="call_1",
                function=SimpleNamespace(name="read", arguments='{"file_path"'),
            )],
        )),
        _chunk(SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                index=0, id=None,
                function=SimpleNamespace(name=None, arguments=': "/tmp/x"}'),
            )],
        )),
        _chunk(SimpleNamespace(content=None, tool_calls=None), finish="tool_calls"),
    ]

    async def create(**kwargs):
        async def _gen():
            for c in chunks:
                yield c
        return _gen()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(
        stream_mod._provider, "build_client",
        lambda key, base_url=None, proxy=None: fake_client,
    )

    async def main():
        evs = []
        async for ev in stream_mod.stream_simple(
            Model(id="m", base_url="http://y"),
            Context(system_prompt="s",
                    messages=[UserMessage(content="hi", timestamp=1.0)]),
        ):
            evs.append(ev)
        return evs

    evs = asyncio.run(main())
    msg = [e for e in evs if e.type == "done"][0].message
    tc = [c for c in msg.content if c.get("type") == "toolCall"][0]
    assert tc["arguments"] == {"file_path": "/tmp/x"}
    assert "args_error" not in tc


# ── agent_loop：args_error 拒执行 + 静默 no-op nudge ─────────


def test_prepare_tool_call_refuses_args_error():
    from connectclaw.agent import agent_loop
    from connectclaw.agent.types import AgentContext, AgentLoopConfig, AgentTool
    from connectclaw.provider.types import Model

    class _FakeTool(AgentTool):
        name = "fake"

        async def execute(self, tool_call_id, params, signal=None, on_update=None):
            raise AssertionError("must not execute")

    ctx = AgentContext(messages=[], tools=[_FakeTool()])
    tc = {"id": "c1", "name": "fake", "arguments": {},
          "args_error": "参数在流式传输中被截断"}
    with pytest.raises(RuntimeError, match="截断"):
        asyncio.run(agent_loop._prepare_tool_call(
            ctx, tc, AgentLoopConfig(model=Model(id="m")), None))


def _silent_stream_fn(state):
    def _stream_fn(model, context, **kwargs):
        async def _gen():
            state["n"] += 1
            if state["n"] >= 2:
                return
            m = _assistant([{"type": "thinking", "thinking": "hmm"}], state["n"])
            yield StreamEvent(type="start",
                              partial=_assistant([], state["n"]))
            yield StreamEvent(type="thinking_delta", delta="hmm",
                              content_index=0, partial=m)
            yield StreamEvent(type="done", message=m)
        return _gen()
    return _stream_fn


def _assistant(blocks, ts):
    from connectclaw.provider.types import AssistantMessage
    return AssistantMessage(content=blocks, model="m",
                            stop_reason="stop", timestamp=float(ts))


from connectclaw.provider.types import StreamEvent, UserMessage  # noqa: E402


def test_loop_nudges_silent_noop_then_resumes(monkeypatch):
    from connectclaw.agent import agent_loop
    from connectclaw.agent.types import AgentContext, AgentLoopConfig
    from connectclaw.provider.types import AssistantMessage, Model

    state = {"n": 0}

    def _stream_fn(model, context, **kwargs):
        async def _gen():
            state["n"] += 1
            if state["n"] == 1:
                m = _assistant([{"type": "thinking", "thinking": "hmm"}], 1.0)
                yield StreamEvent(type="start", partial=_assistant([], 1.0))
                yield StreamEvent(type="thinking_delta", delta="hmm",
                                  content_index=0, partial=m)
                yield StreamEvent(type="done", message=m)
            else:
                m = _assistant([{"type": "text", "text": "hello"}], 2.0)
                yield StreamEvent(type="start", partial=_assistant([], 2.0))
                yield StreamEvent(type="text_delta", delta="hello",
                                  content_index=0, partial=m)
                yield StreamEvent(type="done", message=m)
        return _gen()

    monkeypatch.setattr(agent_loop, "stream_simple", _stream_fn)
    ctx = AgentContext(messages=[UserMessage(content="hi", timestamp=0.0)])

    collected = []

    def _collect(e):
        collected.append(e)

    import asyncio as _a

    async def _emit(e):
        _collect(e)

    async def run():
        await agent_loop._run_loop(ctx, [],
                                   AgentLoopConfig(model=Model(id="m")),
                                   None, _emit)
    _a.run(run())
    assert state["n"] == 2
    nudges = [m for m in ctx.messages
              if isinstance(m, UserMessage) and "系统提示" in str(m.content)]
    assert len(nudges) == 1
    last = ctx.messages[-1]
    assert isinstance(last, AssistantMessage)
    assert any(b.get("type") == "text" and b.get("text", "").strip()
               for b in last.content)


def test_loop_gives_up_after_two_nudges(monkeypatch):
    from connectclaw.agent import agent_loop
    from connectclaw.agent.types import AgentContext, AgentLoopConfig
    from connectclaw.provider.types import Model

    state = {"n": 0}
    monkeypatch.setattr(agent_loop, "stream_simple", _silent_stream_fn(state))
    ctx = AgentContext(messages=[UserMessage(content="hi", timestamp=0.0)])

    async def _emit(e):
        pass

    async def run():
        await agent_loop._run_loop(ctx, [],
                                   AgentLoopConfig(model=Model(id="m")),
                                   None, _emit)

    asyncio.run(run())
    # 1 次原始 + 2 次 nudge 后的重试；第 3 次仍静默 → 放行收尾
    assert state["n"] == 3
    nudges = [m for m in ctx.messages
              if isinstance(m, UserMessage) and "系统提示" in str(m.content)]
    assert len(nudges) == 2
