"""压缩后上下文组装：摘要 + （squash 出来的）注入状态块 + 保留的近期消息。"""

from __future__ import annotations

from connectclaw.agent.harness.session import (
    CompactionEntry,
    ContextEntry,
    MessageEntry,
    SessionHeader,
    build_session_context,
)


def _msg(eid: str, role: str, text: str) -> MessageEntry:
    return MessageEntry(id=eid, parent_id=None, timestamp="2026-09-01T00:00:00Z",
                        message={"role": role, "content": text})


def _entries():
    return [
        SessionHeader(id="s1", created_at="2026-09-01T00:00:00Z", cwd="/tmp"),
        _msg("u1", "user", "<remembered-context>\n- 甲\n</remembered-context>\n\n问题一"),
        _msg("a1", "assistant", "回答一"),
        _msg("u2", "user", "问题二"),
        _msg("a2", "assistant", "回答二"),
    ]


def test_compaction_inserts_summary_then_merged_context():
    entries = _entries() + [
        CompactionEntry(
            id="c1", parent_id="a2", timestamp="2026-09-01T01:00:00Z",
            summary="## 目标\n做事", first_kept_entry_id="u2", tokens_before=1234,
            merged_context="<remembered-context>\n- [2026-07-27 · 1.00] 甲\n</remembered-context>",
        ),
        _msg("u3", "user", "问题三"),
    ]
    ctx = build_session_context(entries)
    contents = [getattr(m, "content", "") for m in ctx.messages]

    assert getattr(ctx.messages[0], "summary", "") == "## 目标\n做事"   # 摘要消息用 .summary
    assert "<remembered-context>" in contents[1], "合并状态块紧跟摘要"   # squash 出来的状态
    assert "甲" in contents[1]
    # 被保留的近期消息仍在，且在压缩产物之后
    assert "问题二" in contents[2]
    assert "回答二" in contents[3]
    assert "问题三" in contents[-1]
    # 压缩点之前的消息被替换掉了
    assert not any("问题一" in c for c in contents)
    assert ctx.compaction_summary == "## 目标\n做事"


def test_compaction_without_merged_context_still_works():
    """老会话（压缩条目没有 merged_context）照旧：只插摘要。"""
    entries = _entries() + [
        CompactionEntry(id="c1", parent_id="a2", timestamp="2026-09-01T01:00:00Z",
                        summary="旧摘要", first_kept_entry_id="u2", tokens_before=1),
        _msg("u3", "user", "问题三"),
    ]
    ctx = build_session_context(entries)
    contents = [getattr(m, "content", "") for m in ctx.messages]
    assert getattr(ctx.messages[0], "summary", "") == "旧摘要"
    assert "<remembered-context>" not in contents[0]
    assert "问题二" in contents[1]


# ── 结构化上下文：op 落盘、文本只在边界渲染 ────────────────────
#
# 用户 2026-09-25 定的架构：状态是 Python 对象、可增删改 diff；文本只在发给 API 那一刻
# 渲染，且历史渲染是**冻结**的（前缀缓存严格匹配，历史一改写命中就断）。

def test_context_entry_renders_before_user_message():
    from connectclaw.injection import OP_CATALOG_SET, OP_MEMORY_ADD

    ops = [
        {"op": OP_MEMORY_ADD, "id": "m1", "line": "- [2026-09-24 · 1.00] 甲", "content": "甲"},
        {"op": OP_CATALOG_SET, "text": "## 可运行的 agents\n- search"},
    ]
    entries = [
        SessionHeader(id="s1", created_at="2026-09-01T00:00:00Z", cwd="/tmp"),
        ContextEntry(id="c1", parent_id=None, timestamp="2026-09-01T00:00:00Z",
                     ops=ops, note="增量"),
        _msg("u1", "user", "用户的真实问题"),     # 用户消息只装用户的话
        _msg("a1", "assistant", "回答"),
    ]
    ctx = build_session_context(entries)
    contents = [getattr(m, "content", "") for m in ctx.messages]

    assert "<remembered-context>" in contents[0]
    assert "甲" in contents[0] and "search" in contents[0]
    assert contents[1] == "用户的真实问题", "用户消息里不再混着注入文本"
    assert len(ctx.messages) == 3


def test_empty_ops_render_nothing():
    entries = [
        SessionHeader(id="s1", created_at="2026-09-01T00:00:00Z", cwd="/tmp"),
        ContextEntry(id="c1", parent_id=None, timestamp="2026-09-01T00:00:00Z", ops=[]),
        _msg("u1", "user", "问题"),
    ]
    ctx = build_session_context(entries)
    assert [getattr(m, "content", "") for m in ctx.messages] == ["问题"]


def test_compaction_snapshot_entry_renders_and_folds_at_cut():
    """压缩条目里的**结构化快照**：既要在上下文里渲染出来，也要在折叠时按切口生效。"""
    from connectclaw.injection import fold_ops

    from connectclaw.injection import render_ops

    snapshot = {"op": "state_snapshot",
                "memory": {"m1": {"line": "- [2026-07-01 · 1.00] 远古", "content": "远古"}},
                "catalog": ""}
    # 生产里是 prepare_compaction 生成：文本渲染（冻结，发给模型用）+ 结构化快照（折叠用）
    snapshot_text = render_ops([snapshot], incremental=False)
    entries = [
        SessionHeader(id="s1", created_at="2026-09-01T00:00:00Z", cwd="/tmp"),
        ContextEntry(id="c0", parent_id=None, timestamp="2026-09-01T00:00:00Z",
                     ops=[{"op": "memory_add", "id": "m9", "line": "- 被压缩掉的", "content": "被压缩掉的"}]),
        _msg("u1", "user", "早期问题"),
        CompactionEntry(id="cp1", parent_id="u1", timestamp="2026-09-01T01:00:00Z",
                        summary="摘要", first_kept_entry_id="u2", tokens_before=100,
                        merged_context=snapshot_text, context_state=snapshot),
        _msg("u2", "user", "保留区问题"),
        ContextEntry(id="c2", parent_id="u2", timestamp="2026-09-01T02:00:00Z",
                     ops=[{"op": "memory_add", "id": "m2", "line": "- 保留区新增", "content": "保留区新增"}]),
    ]
    ctx = build_session_context(entries)
    contents = [getattr(m, "content", "") for m in ctx.messages]
    assert "远古" in contents[0] or any("远古" in c for c in contents), "快照要渲染进上下文"
    assert not any("被压缩掉的" in c for c in contents), "被压缩掉的批次不该再出现"

    # 折叠语义：快照作基、被压缩掉的批次丢掉、保留区的 op 叠在上面
    compacted = [{"op": "memory_add", "id": "m9", "line": "- 被压缩掉的", "content": "被压缩掉的"}]
    kept_ops = [{"op": "memory_add", "id": "m2", "line": "- 保留区新增", "content": "保留区新增"}]
    state = fold_ops([compacted, [snapshot], kept_ops])
    assert set(state.memory) == {"m1", "m2"}, "被压缩掉的 m9 不该在折叠结果里"


def test_harness_folds_context_state_at_compaction_cut():
    """端到端：真会话文件 → harness.context_state() 折叠。

    这段是这次重构的关键点：压缩条目在**文件里排在保留区之后**，直接按文件顺序折叠会把
    保留区的 op 先作用、再被快照清掉。所以折叠必须按 `first_kept_entry_id` 切口。
    """
    import asyncio
    import json
    import tempfile
    from pathlib import Path

    from connectclaw.agent.harness.agent_harness import AgentHarness
    from connectclaw.agent.harness.session import JsonlSessionStorage, _entry_to_dict
    from connectclaw.provider.types import Model

    mem = lambda mid, content: {"op": "memory_add", "id": mid, "line": f"- {content}", "content": content}  # noqa: E731
    entries = [
        SessionHeader(id="s1", created_at="2026-09-01T00:00:00Z", cwd="/tmp"),
        ContextEntry(id="c0", parent_id=None, timestamp="2026-09-01T00:00:00Z", ops=[mem("m9", "早期")]),
        _msg("u1", "user", "早期问题"),
        CompactionEntry(id="cp1", parent_id="u1", timestamp="2026-09-01T01:00:00Z",
                        summary="摘要", first_kept_entry_id="u2", tokens_before=100,
                        merged_context="x", context_state={
                            "op": "state_snapshot",
                            "memory": {"m1": {"line": "- 远古", "content": "远古"}},
                            "catalog": "## 可运行的 agents",
                        }),
        _msg("u2", "user", "保留区问题"),
        ContextEntry(id="c2", parent_id="u2", timestamp="2026-09-01T02:00:00Z", ops=[mem("m2", "保留区新增")]),
        _msg("u3", "user", "最新问题"),
    ]

    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for e in entries:
                    fh.write(json.dumps(_entry_to_dict(e), default=str, ensure_ascii=False) + "\n")
            storage = await JsonlSessionStorage.open(str(path))
            harness = AgentHarness(session=storage, model=Model(id="test"), system_prompt="", tools=[])
            return await harness.context_state()

    state = asyncio.run(run())
    assert set(state.memory) == {"m1", "m2"}, f"切口折叠错了：{sorted(state.memory)}"
    assert "m9" not in state.memory, "被压缩掉的条目不该留在状态里"
    assert state.catalog == "## 可运行的 agents"
