"""Compaction hardening tests.

Covers the fixes added to context compaction:
  1. Real-token (tiktoken) estimation instead of the chars/4 heuristic that
     under-counted CJK 2-4x and could push the kept-recent window over budget.
  2. Benefit guard: compaction with nothing to condense returns None instead of
     prepending a pointless summary ("compact → still over → never shrinks").
  3. Budget-fit keep window: with context_window supplied, the cut is reduced so
     the post-compaction context (kept + summary) fits window - reserve.
  4. Summary/file-ops caps and budget-aware <conversation> serialization.
"""

from __future__ import annotations

from connectclaw.agent.harness.compaction import (
    CompactionSettings,
    _cap_summary,
    _extract_file_ops,
    _serialize,
    count_tokens,
    estimate_tokens,
    find_cut_point,
    prepare_compaction,
    strip_tool_call_artifacts,
)


def _mk(prefix: str, n: int, content: str) -> list[dict]:
    return [
        {"type": "message", "id": f"{prefix}{i}",
         "message": {"role": "user" if i % 2 == 0 else "assistant", "content": content}}
        for i in range(n)
    ]


WINDOW, RESERVE, KEEP = 65536, 16384, 20000
SETTINGS = CompactionSettings(enabled=True, reserve_tokens=RESERVE, keep_recent_tokens=KEEP)


# ── 1. Real-token estimation ─────────────────────────────────


def test_cjk_no_longer_undercounted():
    sample = "中文字符内容用来测试编码器是否准确。"
    est = count_tokens(sample)
    chars4 = len(sample) // 4
    # Real tokenizer should NOT undercount CJK the way chars/4 did.
    assert est >= chars4
    assert est >= 1


def test_cjk_keep_window_real_tokens():
    entries = _mk("m", 60, "中文字符内容" * 60)
    cut = find_cut_point(entries, 0, len(entries), KEEP)
    kept = entries[cut.first_kept_entry_index:]
    real_kept = sum(estimate_tokens(e["message"]) for e in kept)
    # Kept window must be measured in real tokens, not blown up 2-4x past budget.
    assert real_kept <= KEEP * 1.6


def test_estimate_tokens_always_positive():
    assert estimate_tokens({"role": "user", "content": ""}) >= 1
    assert estimate_tokens({"role": "bashExecution", "command": "", "output": ""}) >= 1


def test_estimate_tokens_tolerates_non_dict_content_blocks():
    # user / toolResult content 里的非 dict 块不应崩溃（与 assistant 分支一致）
    assert estimate_tokens({"role": "user", "content": ["plain-string-block"]}) >= 1
    assert estimate_tokens({"role": "user", "content": [{"text": "ok"}, "raw"]}) >= 1
    assert estimate_tokens({"role": "toolResult", "content": [{"text": "ok"}, 42]}) >= 1


# ── 1b. Image blocks (use-then-drop policy) ─────────────────────


def test_estimate_tokens_counts_image_blocks():
    # image/image_ref blocks count at a fixed per-image cost (not chars/4),
    # so a branch with images must price them in.
    img = {"type": "image_ref", "id": "x", "path": "/none", "mime_type": "image/png", "size": 100 * 1024}
    est = estimate_tokens({"role": "user", "content": [{"type": "text", "text": "hi"}, img]})
    assert est >= 800  # 800 = image tier for 100KB
    est_r = estimate_tokens({"role": "toolResult", "content": [img]})
    assert est_r >= 800


def test_serialize_image_placeholder():
    msgs = [{
        "role": "user",
        "content": [{"type": "text", "text": "look at this"},
                    {"type": "image_ref", "id": "abc", "path": "/none",
                     "mime_type": "image/png", "size": 128 * 1024}],
    }]
    out = _serialize(msgs)
    assert "[image abc image/png 128KB]" in out
    assert "look at this" in out


# ── 2. Benefit guard ──────────────────────────────────────────


def test_refuses_pointless_compaction():
    tiny = [{"type": "message", "id": "t1", "message": {"role": "user", "content": "hi"}},
            {"type": "message", "id": "t2", "message": {"role": "assistant", "content": "yo"}}]
    # Nothing meaningful to condense → refuse, do NOT emit a summary.
    assert prepare_compaction(list(tiny), SETTINGS, context_window=WINDOW) is None


def test_refuses_when_last_entry_is_compaction():
    entries = _mk("m", 30, "hello")
    entries.append({"type": "compaction", "id": "c1", "summary": "s", "first_kept_entry_id": "m0"})
    assert prepare_compaction(entries, SETTINGS, context_window=WINDOW) is None


# ── 3. Budget-fit keep window ─────────────────────────────────


def test_post_compact_fits_budget():
    # Window large enough that the keep-cut must condense real messages.
    entries = _mk("m", 400, "中文内容" * 30)
    prep = prepare_compaction(list(entries), SETTINGS, context_window=WINDOW)
    assert prep is not None
    # The budget-fit keeps the post-compaction estimate under window - reserve.
    assert prep.post_compact_estimate <= WINDOW - RESERVE
    assert prep.fits_budget is True
    assert prep.liberated_tokens > 0
    assert prep.messages_to_summarize


def test_refuses_small_context_when_not_over_budget():
    # A genuinely small conversation has nothing to condense → refuse.
    entries = _mk("m", 20, "这是一个比较长的中文会话内容用于触发压缩。")
    assert prepare_compaction(list(entries), SETTINGS, context_window=WINDOW) is None


def test_liberates_something_when_over_budget():
    # Content so large that the recent window alone would blow the budget.
    big = _mk("m", 300, "中" * 2000)
    prep = prepare_compaction(list(big), SETTINGS, context_window=WINDOW)
    assert prep is not None
    assert prep.liberated_tokens > 0
    assert prep.post_compact_estimate <= WINDOW - RESERVE


# ── 4. Summary / serialization caps ───────────────────────────


def test_cap_summary_bounds_tokens():
    capped = _cap_summary("x" * 50_000, 2048)
    assert count_tokens(capped) <= 2300  # budget + small note
    assert "truncated" in capped


def test_serialize_drops_oldest_to_fit_budget():
    msgs = [{"role": "user", "content": "a" * 400}] * 200
    out = _serialize(msgs, max_tokens=1000)
    assert count_tokens(out) <= 1500
    assert "dropped to fit" in out


def test_serialize_without_budget_keeps_all():
    msgs = [{"role": "user", "content": "a" * 400}] * 10
    out = _serialize(msgs)
    assert "dropped to fit" not in out
    assert "[" in out  # lines present

# ── 压缩时 squash 注入（git 合并语义）────────────────────────

def _turn(lines: str, *, ts: str) -> list[dict]:
    """一轮：`context` 条目（结构化 op）+ 用户消息（长文本，让窗口真的需要压缩）。

    新格式下注入是独立条目、用户消息是干净的——所以窗口大小由消息决定，
    op 只是旁边挂着的一批增量。
    """
    h = abs(hash(lines + ts)) % 100000
    return [
        {"type": "context", "id": f"c{h}", "parent_id": None, "timestamp": ts,
         "ops": [{"op": "memory_add", "id": f"m{h}", "line": lines, "content": lines}],
         "note": "增量"},
        {"type": "message", "id": f"u{h}", "parent_id": None, "timestamp": ts,
         "message": {"role": "user", "content": "继续" + "长内容" * 120}},
    ]


def _region() -> list[dict]:
    rows: list[dict] = [*_turn("- [2026-07-27 · 1.00] 甲", ts="2026-09-01T00:00:00Z")]
    for i in range(40):
        rows.extend(_turn(f"- [2026-08-01 · 0.80] 记忆{i}", ts=f"2026-09-01T00:{i:02d}:30Z"))
    return rows


def test_prepare_compaction_squashes_injections():
    """区域内散落的注入 op 要合并成一个当前状态（结构化快照 + 冻结文本），随摘要进上下文。"""
    prep = prepare_compaction(_region(), CompactionSettings(), context_window=32768)
    assert prep is not None
    assert "<remembered-context>" in prep.merged_context
    assert "甲" in prep.merged_context
    assert "压缩后合并的记忆状态" in prep.merged_context
    # 结构化快照同时产出（后续折叠不必再解析文本）
    assert prep.context_state is not None
    assert prep.context_state["op"] == "state_snapshot"
    assert "甲" in str(prep.context_state["memory"])


def test_prepare_compaction_folds_previous_snapshot():
    """上一轮压缩的快照是当前上下文的一部分，这次压缩必须把它并进来。"""
    snapshot = {"op": "state_snapshot",
                "memory": {"m1": {"line": "- [2026-07-01 · 1.00] 远古记忆", "content": "远古记忆"}},
                "catalog": ""}
    rows: list[dict] = [{
        "type": "compaction", "id": "cp0", "timestamp": "2026-08-01T00:00:00Z",
        "summary": "早期摘要", "first_kept_entry_id": "u-first",
        "tokens_before": 1000, "merged_context": "x", "context_state": snapshot,
    }]
    rows.extend(_region())
    prep = prepare_compaction(rows, CompactionSettings(), context_window=32768)
    assert prep is not None
    assert "远古记忆" in prep.merged_context, "上一轮的快照必须被折进新的合并结果"
    assert "远古记忆" in str(prep.context_state["memory"])


# ── 5. Summarizer output sanitization & file-ops extraction ──


def test_strip_tool_call_artifacts_dots_block():
    # 2026-09-25 真实事故：dots3 总结模型吐了原生调用语法混进摘要
    dirty = (
        "\n\n\n\n<dots_function_call> <tool_calls> <invoke name=\"search\"> "
        "<parameter name=\"query\">Jev TypeSafe AI RLCD training method how trained</parameter> "
        "<parameter name=\"topn\">5</parameter> </invoke> </tool_calls>\n\nafter"
    )
    assert strip_tool_call_artifacts(dirty) == "after"


def test_strip_tool_call_artifacts_unclosed_block():
    # 没有闭合标签时切到字符串末尾，避免残段混进摘要
    assert strip_tool_call_artifacts("summary text\n<tool_call>{\"name\": \"search\"}") == "summary text"


def test_strip_tool_call_artifacts_keeps_normal_text():
    text = "## Goal\nDo the thing.\n- step 1 <b>not a tool call</b>"
    assert strip_tool_call_artifacts(text) == text


def test_extract_file_ops_skips_none_and_empty_details():
    # details=null 与 details={} 都不能产出字面量 "None" 条目
    msgs = [
        {"role": "toolResult", "tool_name": "web_search", "details": None},
        {"role": "toolResult", "tool_name": "write", "details": {}},
        {"role": "toolResult", "tool_name": "edit", "details": {"file_path": "/tmp/a.py"}},
    ]
    ops = _extract_file_ops(msgs)
    assert ops.edited == {"/tmp/a.py"}
    assert "None" not in ops.edited


def test_extract_file_ops_routes_read_and_write():
    msgs = [
        {"role": "toolResult", "tool_name": "read", "details": {"path": "/tmp/in.txt"}},
        {"role": "toolResult", "tool_name": "write", "details": {"path": "/tmp/out.txt"}},
        {"role": "assistant", "details": {"path": "/tmp/nope"}},  # 非 toolResult 忽略
        {"role": "toolResult", "tool_name": "web_search", "details": {"query": "x"}},  # 无路径忽略
    ]
    ops = _extract_file_ops(msgs)
    assert ops.read == {"/tmp/in.txt"}
    assert ops.edited == {"/tmp/out.txt"}
