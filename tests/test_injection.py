"""注入的结构化模型（op / 折叠 / 渲染）。

2026-09-25 起注入不再"渲染成文本拼进用户消息"，而是独立的 `context` 条目（op）。
旧格式（文本拼在消息里）已由 scripts/migrate_sessions_to_structured.py 一次性迁移，
所以这里不再有"解析旧文本"的测试。

架构要点：
* op 里带着**当时渲染好的行文本** → 历史渲染冻结（前缀缓存要求）；
* 增量判断在对象层面做，只按**内容**比对（渲染文本里的强度每轮都在动）；
* "已经注入过什么"靠 fold 会话推导出来，不是内存账本 → 重启不重发。

（历史绝不被改写这条也有测试：见下面 test_history_is_never_rewritten。）
"""

from __future__ import annotations

from connectclaw.agent.harness.messages import convert_to_llm
from connectclaw.agent.types import BashExecutionMessage, CompactionSummaryMessage
from connectclaw.injection import (
    OP_CATALOG_SET,
    OP_MEMORY_ADD,
    OP_MEMORY_FORGET,
    ContextState,
    fold_ops,
    render_ops,
)
from connectclaw.provider.types import AssistantMessage, UserMessage


USER_TEXT = "帮我查查jev的情况"


def test_history_is_never_rewritten():
    """历史消息必须原样透传——上游严格按请求前缀缓存，改写历史 = 命中在改写点断开。

    2026-09-24 实测教训：曾经把历史里的注入降级成常量占位符（想清理窗口），结果
    每轮多付一整轮未命中（平均命中率 77%→61%，短会话 71%→10%）。窗口要干净得靠
    结构化 + 按需注入，而不是事后改写历史。
    """
    history = UserMessage(content="历史消息", timestamp=1.0)
    current = UserMessage(content="本轮消息", timestamp=2.0)
    assistant = AssistantMessage(content=[{"type": "text", "text": "回答"}])
    out = convert_to_llm([history, assistant, current])

    assert out[0] is history and out[2] is current
    assert out[1] is assistant


def test_list_shaped_user_content_passes_through():
    history = UserMessage(content=[
        {"type": "text", "text": "看这张图"},
        {"type": "image_ref", "id": "x", "path": "/tmp/a.png", "mime_type": "image/png", "size": 1},
    ], timestamp=1.0)
    out = convert_to_llm([history, AssistantMessage(content=[])])
    assert out[0].content == history.content


def test_other_roles_still_converted():
    out = convert_to_llm([
        BashExecutionMessage(command="ls", output="a\nb", timestamp=1.0),
        CompactionSummaryMessage(summary="## 目标\n做事", tokens_before=10, timestamp=2.0),
        UserMessage(content=USER_TEXT, timestamp=3.0),
    ])
    assert "<bash-output command=\"ls\">" in out[0].content
    assert out[1].content.startswith("<summary>")
    assert out[2].content == USER_TEXT


# ── 按需注入：结构化 op（Python 对象，不是文本）───────────────
#
# 用户 2026-09-25 定的架构：状态用对象维护，文本只在发给 API 那一刻渲染。所以增量
# 判断产出的是 op（可 diff / 可合并），而不是拼好的文本；"已经注入过什么"来自
# **从会话折叠出来的状态**，不再依赖内存账本（这才是"重启不重发"的正解）。

def test_memory_delta_emits_ops_only_for_new_or_changed():
    state = ContextState()
    items = [("m1", "- [2026-07-27 · 1.00] 甲", "甲"),
             ("m2", "- [2026-07-27 · 1.00] 乙", "乙")]

    ops, forgotten = state.memory_delta(items)
    assert [o["op"] for o in ops] == [OP_MEMORY_ADD, OP_MEMORY_ADD]
    assert [o["id"] for o in ops] == ["m1", "m2"]
    assert forgotten == []

    state.apply(ops)                       # 折叠进状态
    again, _ = state.memory_delta(items)   # 同样的条目再来一遍
    assert again == [], "这就是「同一句话被重复注入 34 次」的解法"

    # 强度变了（confirm_usage / 衰减每轮都在动）→ 不算变化，不该重发
    restamped = [("m1", "- [2026-09-24 · 0.60] 甲", "甲"),
                 ("m2", "- [2026-09-24 · 0.90] 乙", "乙")]
    assert state.memory_delta(restamped)[0] == []

    # 内容被更正 → 必须重新注入
    corrected = [("m1", "- [2026-09-24 · 0.60] 甲（已更正）", "甲（已更正）"),
                 ("m2", "- [2026-09-24 · 0.90] 乙", "乙")]
    ops, _ = state.memory_delta(corrected)
    assert len(ops) == 1 and "已更正" in ops[0]["line"]


def test_memory_delta_emits_forget_op():
    state = ContextState()
    state.apply(state.memory_delta([("m1", "- 甲", "甲"), ("m2", "- 乙", "乙")])[0])

    ops, forgotten = state.memory_delta([("m2", "- 乙", "乙")], alive=lambda mid: mid == "m2")
    assert ops == []
    assert [o["op"] for o in forgotten] == [OP_MEMORY_FORGET]
    assert forgotten[0]["id"] == "m1"


def test_memory_delta_ignores_missing_when_no_alive_check():
    state = ContextState()
    state.apply(state.memory_delta([("m1", "- 甲", "甲")])[0])
    assert state.memory_delta([]) == ([], [])


def test_catalog_delta_emits_op_only_on_change():
    state = ContextState()
    ops = state.catalog_delta("## 可运行的 agents\n- a")
    assert [o["op"] for o in ops] == [OP_CATALOG_SET]
    state.apply(ops)
    assert state.catalog_delta("## 可运行的 agents\n- a") == []
    assert state.catalog_delta("## 可运行的 agents\n- a\n- b") != []
    assert state.catalog_delta("") == []


def test_render_ops_produces_incremental_block():
    ops = [{"op": OP_MEMORY_ADD, "id": "m1", "line": "- [2026-07-27 · 1.00] 甲", "content": "甲"},
           {"op": OP_MEMORY_FORGET, "id": "m9"},
           {"op": OP_CATALOG_SET, "text": "## 可运行的 agents\n- search"}]
    text = render_ops(ops)
    assert text.startswith("<remembered-context>")
    assert "记忆更新" in text                      # 增量表头
    assert "甲" in text and "search" in text
    assert "m9" not in text                        # forget 不该渲染成内容


def test_fold_ops_is_ordered_and_snapshot_replaces():
    add_a = [{"op": OP_MEMORY_ADD, "id": "a", "line": "- 甲", "content": "甲"}]
    add_b = [{"op": OP_MEMORY_ADD, "id": "b", "line": "- 乙", "content": "乙"}]
    forget_a = [{"op": OP_MEMORY_FORGET, "id": "a"}]

    state = fold_ops([add_a, add_b])
    assert set(state.memory) == {"a", "b"}
    state = fold_ops([add_a, add_b, forget_a])
    assert set(state.memory) == {"b"}

    # 压缩快照 = 整份状态替换（折叠时它会被放在保留区之前）
    snap = ContextState()
    snap.apply(add_a)
    state = fold_ops([add_a, add_b, snap.snapshot_ops()])
    assert set(state.memory) == {"a"}, "快照是「当时状态」，不是并集"


def test_fold_ops_is_ordered_and_snapshot_replaces():
    add_a = [{"op": OP_MEMORY_ADD, "id": "a", "line": "- 甲", "content": "甲"}]
    add_b = [{"op": OP_MEMORY_ADD, "id": "b", "line": "- 乙", "content": "乙"}]
    forget_a = [{"op": OP_MEMORY_FORGET, "id": "a"}]

    state = fold_ops([add_a, add_b])
    assert set(state.memory) == {"a", "b"}
    state = fold_ops([add_a, add_b, forget_a])
    assert set(state.memory) == {"b"}

    # 压缩快照 = 整份状态替换（折叠时它会被放在保留区之前）
    snap = ContextState()
    snap.apply(add_a)
    state = fold_ops([add_a, add_b, snap.snapshot_ops()])
    assert set(state.memory) == {"a"}, "快照是「当时状态」，不是并集"


