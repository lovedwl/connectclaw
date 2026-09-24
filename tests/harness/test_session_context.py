"""压缩后上下文组装：摘要 + （squash 出来的）注入状态块 + 保留的近期消息。"""

from __future__ import annotations

from connectclaw.agent.harness.session import (
    CompactionEntry,
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
