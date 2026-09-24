"""注入块的识别与剥离。

背景（2026-09-24 实测）：每轮对话都会把「记忆 / RAG 文档 / agents 与工具清单」
拼在用户消息前面，随消息一起落盘。168 个用户轮 100% 带记忆块，同一句话平均
被重复注入 34 次，注入总量 297k 字；配置类旧说法（「RAG 已启用」）出现 84 次，
而当天写的更正只出现 1 次。这些副本原样进上下文，既费 token 又让过时说法与
最新事实同框冲突。

修法：历史轮次的注入块降级成常量占位符，只保留最近一条用户消息里的那份。
"""

from __future__ import annotations

from connectclaw.agent.harness.messages import convert_to_llm
from connectclaw.agent.types import BashExecutionMessage, CompactionSummaryMessage
from connectclaw.injection import (
    INJECTION_PLACEHOLDER,
    InjectionLedger,
    LedgerRegistry,
    merge_injections,
    has_injections,
    split_injections,
    strip_injections,
    strip_message_content,
)
from connectclaw.provider.types import AssistantMessage, ToolResultMessage, UserMessage

MEMORY_BLOCK = """<remembered-context>
(Things you know from past interactions. 每条前缀 [日期 · 强度]：日期是记录时间，强度 0~1 是可信度)
- [2026-07-27 · 1.00] Vision模型配置为 qwen3.6-chat，RAG已启用
- [2026-09-24 · 1.00] 当前模型路由已直连 dots.ai，不走 USTC 代理
</remembered-context>"""

AGENTS_SECTION = """## 可运行的 agents(用 `agents(action="run", agent="<名>")` 调用)
- search — 网络搜索专家，执行搜索并返回精心整理的结果摘要
- reviewer — 代码审查"""

AGENTS_EMPTY = """## 可运行的 agents
(暂无。用 `agents(action="create", ...)` 造一个,当轮即可 run)"""

TOOLS_SECTION = """## 可授权给子 agent 的工具(create 时写进 `tools:[]`)
read, write, hash_read, hash_edit, bash, web_search, memory"""

RAG_BLOCK = """<retrieved-documents>
## Relevant Documentation

片段一
---
片段二
</retrieved-documents>"""

USER_TEXT = "帮我查查jev的情况"


def injected(*blocks: str, user_text: str = USER_TEXT) -> str:
    return "\n\n".join(blocks) + "\n\n" + user_text


# ── 识别 ─────────────────────────────────────────────────────

def test_has_injections_detects_real_layout():
    assert has_injections(injected(MEMORY_BLOCK, AGENTS_SECTION, TOOLS_SECTION))
    assert not has_injections(USER_TEXT)
    assert not has_injections("")


def test_split_separates_all_blocks_from_user_text():
    injected_text = injected(MEMORY_BLOCK, RAG_BLOCK, AGENTS_SECTION, TOOLS_SECTION)
    blocks, rest = split_injections(injected_text)

    assert rest == USER_TEXT
    assert "<remembered-context>" in blocks
    assert "<retrieved-documents>" in blocks
    assert "可运行的 agents" in blocks
    assert "可授权给子 agent 的工具" in blocks
    # 注入段里不该混进用户的话
    assert USER_TEXT not in blocks


def test_split_handles_empty_agents_variant():
    blocks, rest = split_injections(injected(AGENTS_EMPTY, TOOLS_SECTION))
    assert rest == USER_TEXT
    assert "暂无" in blocks


def test_plain_text_untouched_including_indentation():
    text = "    缩进开头的正文\n第二行"
    assert split_injections(text) == ("", text)
    assert strip_injections(text) == text


def test_empty_and_none_like_inputs():
    assert split_injections("") == ("", "")
    assert strip_injections("") == ""


# ── 剥离 ─────────────────────────────────────────────────────

def test_strip_replaces_with_constant_placeholder():
    stripped = strip_injections(injected(MEMORY_BLOCK, AGENTS_SECTION))
    assert stripped.startswith(INJECTION_PLACEHOLDER)
    assert stripped.endswith(USER_TEXT)
    # 原始内容不再出现
    assert "RAG已启用" not in stripped
    assert "search —" not in stripped


def test_strip_is_idempotent_and_byte_stable():
    once = strip_injections(injected(MEMORY_BLOCK, AGENTS_SECTION))
    twice = strip_injections(once)
    assert once == twice
    # 两份不同内容的注入，剥完必须完全一致（历史前缀才稳定可缓存）
    other = strip_injections(
        injected(MEMORY_BLOCK.replace("Vision模型配置", "别的记忆"), TOOLS_SECTION)
    )
    assert once.split(USER_TEXT)[0] == other.split(USER_TEXT)[0]


def test_only_injection_yields_placeholder_only():
    stripped = strip_injections(MEMORY_BLOCK)
    assert stripped == INJECTION_PLACEHOLDER


def test_strip_keeps_text_after_injection_intact():
    text = injected(MEMORY_BLOCK, user_text="第二行有内容\n第三行也有")
    assert strip_injections(text).endswith("第二行有内容\n第三行也有")


# ── 两种 content 形状 ────────────────────────────────────────

def test_strip_message_content_string():
    assert strip_injections(injected(MEMORY_BLOCK)) == strip_message_content(injected(MEMORY_BLOCK))


def test_strip_message_content_list_keeps_image_blocks():
    content = [
        {"type": "text", "text": injected(MEMORY_BLOCK, user_text="看这张图")},
        {"type": "image_ref", "id": "abc", "path": "/tmp/x.png", "mime_type": "image/png", "size": 10},
    ]
    out = strip_message_content(content)
    assert out[0]["text"].startswith(INJECTION_PLACEHOLDER)
    assert out[0]["text"].endswith("看这张图")
    assert out[1] == content[1]  # 图片块原样保留


def test_strip_message_content_passes_through_unknown_shapes():
    assert strip_message_content(None) is None
    assert strip_message_content(123) == 123


# ── 历史绝不被改写（前缀缓存）────────────────────────────────

def test_history_is_never_rewritten():
    """注入块随用户消息落盘、就是下一轮的固定前缀，**不能改写它**。

    2026-09-24 实测教训：曾经把历史里的注入降级成常量占位符（想清理窗口），结果
    前缀正好在"上一轮"处断开，每轮多付一整轮未命中——平均命中率 77%→61%，
    短会话 71%→10%。窗口要干净得靠按需注入，不是事后改写历史。这条测试把决定钉死。
    """
    messages = [
        UserMessage(content=injected(MEMORY_BLOCK, AGENTS_SECTION), timestamp=1.0),
        AssistantMessage(content=[{"type": "text", "text": "第一次回答"}]),
        ToolResultMessage(tool_call_id="c1", tool_name="bash", content=[{"type": "text", "text": "ok"}]),
        UserMessage(content=injected(MEMORY_BLOCK, AGENTS_SECTION, user_text="再查一次"), timestamp=2.0),
    ]
    out = convert_to_llm(messages)

    assert out[0].content == messages[0].content, "历史用户消息必须原样透传"
    assert out[-1].content == messages[-1].content
    assert out[1] is messages[1]
    assert INJECTION_PLACEHOLDER not in out[0].content


def test_list_shaped_user_content_passes_through():
    history = UserMessage(content=[
        {"type": "text", "text": injected(MEMORY_BLOCK, AGENTS_SECTION)},
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


# ── 按需注入账本 ─────────────────────────────────────────────

def test_memory_delta_only_reports_new_or_changed():
    ledger = InjectionLedger()
    items = [("m1", "- [2026-07-27 · 1.00] 甲", "甲"),
             ("m2", "- [2026-07-27 · 1.00] 乙", "乙")]

    new, forgotten = ledger.memory_delta(items)
    assert new == ["- [2026-07-27 · 1.00] 甲", "- [2026-07-27 · 1.00] 乙"]
    assert forgotten == []

    # 同样的条目再来一遍：什么都不注入（这就是"同一句话被重复注入 34 次"的解法）
    new, forgotten = ledger.memory_delta(items)
    assert new == [] and forgotten == []

    # 强度变了（confirm_usage / 衰减每轮都在动它）→ **不算变化**，不该重发
    restamped = [("m1", "- [2026-09-24 · 0.60] 甲", "甲"),
                 ("m2", "- [2026-09-24 · 0.90] 乙", "乙")]
    new, _ = ledger.memory_delta(restamped)
    assert new == [], "强度波动不该触发重新注入（第一版按渲染文本比对就是这么坏掉的）"

    # 内容被更正 → 必须重新注入
    corrected = [("m1", "- [2026-09-24 · 0.60] 甲（已更正）", "甲（已更正）"),
                 ("m2", "- [2026-09-24 · 0.90] 乙", "乙")]
    new, _ = ledger.memory_delta(corrected)
    assert new == ["- [2026-09-24 · 0.60] 甲（已更正）"]


def test_memory_delta_reports_forgotten_items():
    ledger = InjectionLedger()
    ledger.memory_delta([("m1", "- 甲", "甲"), ("m2", "- 乙", "乙")])

    # m1 被 /forget 或删掉了 → 要告知（"少召回"那一侧）
    new, forgotten = ledger.memory_delta(
        [("m2", "- 乙", "乙")], alive=lambda mid: mid == "m2"
    )
    assert new == []
    assert forgotten == ["- 已遗忘：甲"]
    # 报过就从账本里移除，不要每轮重复报
    assert "m1" not in ledger.memory


def test_memory_delta_ignores_missing_when_no_alive_check():
    ledger = InjectionLedger()
    ledger.memory_delta([("m1", "- 甲", "甲")])
    new, forgotten = ledger.memory_delta([])
    assert new == [] and forgotten == []


def test_catalog_delta_only_on_change():
    ledger = InjectionLedger()
    first = ledger.catalog_delta("## 可运行的 agents\n- a")
    assert first != ""
    assert ledger.catalog_delta("## 可运行的 agents\n- a") == ""
    assert ledger.catalog_delta("## 可运行的 agents\n- a\n- b") != ""
    assert ledger.catalog_delta("") == ""


def test_ledger_registry_is_per_session():
    registry = LedgerRegistry()
    assert registry.for_session("s1") is registry.for_session("s1")
    # 换会话（/new）→ 新账本 → 首轮重新完整注入一次
    assert registry.for_session("s1") is not registry.for_session("s2")
    assert len(registry) == 2
    registry.drop("s1")
    assert len(registry) == 1


# ── 压缩时 squash 注入（git 合并语义）────────────────────────

def _mem(*lines: str) -> str:
    return "<remembered-context>\n(header)\n" + "\n".join(lines) + "\n</remembered-context>"


def test_merge_unions_and_last_write_wins():
    merged = merge_injections([
        _mem("- [2026-07-27 · 1.00] 甲", "- [2026-07-27 · 0.60] 乙"),
        _mem("- [2026-08-01 · 0.90] 丙"),
        _mem("- [2026-08-02 · 0.31] 甲"),   # 同内容再来一次 → 以最后一次为准
    ])
    assert "丙" in merged
    assert "[2026-08-02 · 0.31] 甲" in merged
    assert merged.count("甲") == 1, "同内容只该留一条"


def test_merge_honours_forgotten_lines():
    """`- 已遗忘：X` 等于后来的提交 revert 掉 X —— squash 后 X 不该还在。"""
    merged = merge_injections([
        _mem("- [2026-07-27 · 1.00] 甲", "- [2026-07-27 · 0.60] 乙"),
        _mem("- 已遗忘：乙"),
    ])
    assert "甲" in merged
    assert "乙" not in merged
    assert "已遗忘" not in merged, "撤销行本身不是条目，不该留在状态里"


def test_merge_takes_latest_catalog():
    merged = merge_injections([
        _mem("- 甲") + "\n\n## 可运行的 agents\n- search — 搜索",
        _mem("- 乙") + "\n\n## 可运行的 agents\n- search — 搜索\n- reviewer — 审查",
    ])
    assert "reviewer" in merged
    assert merged.count("可运行的 agents") == 1


def test_merge_is_idempotent_and_handles_empty():
    """合并结果本身也是标准注入块，所以下次压缩可以继续合并它。"""
    once = merge_injections([_mem("- [2026-07-27 · 1.00] 甲")])
    assert merge_injections([once]) == once
    assert merge_injections([]) == ""
    assert merge_injections(["", "   "]) == ""


def test_strip_handles_section_without_trailing_newline():
    """末行没有换行时也要整段吃掉（原来只写 \\n 会漏掉最后一行）。"""
    stripped = strip_injections("## 可运行的 agents\n- item-one\n- item-two")
    assert stripped == INJECTION_PLACEHOLDER
    assert "item-one" not in stripped and "item-two" not in stripped
