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
    compaction_trigger,
    count_tokens,
    estimate_tokens,
    find_cut_point,
    prepare_compaction,
    should_compact,
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


# ── 0. 预算分配：触发线 / 保留窗口 / 压缩产物 ─────────────────
# 400k 窗口 + 64k 输出预留 → 可用 344k；80% 触发（275k），保留最近 20%
# （69k），摘要+快照最多 33k → 压缩后约 102k，离触发线还有 173k。

def test_trigger_is_fraction_of_usable_budget():
    s = CompactionSettings(enabled=True, reserve_tokens=65536, trigger_ratio=0.8)
    usable = 409600 - 65536
    trigger = compaction_trigger(409600, s)
    assert trigger == int(usable * 0.8)
    assert should_compact(trigger, 409600, s) is False
    assert should_compact(trigger + 1, 409600, s) is True


def test_worst_case_post_compaction_has_large_headroom():
    s = CompactionSettings(
        enabled=True, reserve_tokens=65536, trigger_ratio=0.8,
        keep_recent_tokens=68800, summary_max_tokens=24576,
        memory_snapshot_max_tokens=8192,
    )
    trigger = compaction_trigger(409600, s)
    worst_post = (
        s.keep_recent_tokens + s.summary_max_tokens + s.memory_snapshot_max_tokens
    )
    # 压缩后必须远低于触发线，否则工具输出几轮就又顶回去（88 次抖动事故）。
    assert worst_post < trigger * 0.4


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
    # Keep the summary small relative to the condensed region: a compaction that
    # cannot shrink is now refused before it reaches this assert.
    settings = CompactionSettings(
        keep_recent_tokens=4000, summary_max_tokens=1024,
    )
    prep = prepare_compaction(_region(), settings, context_window=32768)
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
    settings = CompactionSettings(
        keep_recent_tokens=4000, summary_max_tokens=1024,
    )
    prep = prepare_compaction(rows, settings, context_window=32768)
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


# ── 6. 压缩产物封顶：摘要 + 记忆快照（2026-10 抖动事故）───────
# 事故形态：摘要上限 = 0.8×reserve ≈ 13k token、记忆快照不封顶（实测 83 条 /
# 16k token），两者都留在压缩后的上下文里 → 压缩完仍有 49k > 触发阈值 49.1k
# → 下一轮立刻再压，一个会话压了 88 次；每次压缩 4-6 分钟，用户看到的就是
# "发消息后迟迟不思考"。


def test_summary_cap_comes_from_settings_not_reserve():
    settings = CompactionSettings(summary_max_tokens=512)
    assert settings.summary_max_tokens == 512
    assert settings.summary_max_tokens < int(settings.reserve_tokens * 0.8)


def test_post_compact_estimate_includes_merged_context():
    settings = CompactionSettings(
        reserve_tokens=4096, keep_recent_tokens=4000,
        summary_max_tokens=1024, memory_snapshot_max_tokens=256,
    )
    prep = prepare_compaction(_region(), settings, context_window=16384)
    assert prep is not None

    merged_tokens = count_tokens(prep.merged_context)
    # 快照按 token 预算裁剪，不再无限膨胀
    assert merged_tokens <= settings.memory_snapshot_max_tokens + 300  # 表头开销
    # 预算估算必须把快照算进去（旧实现完全忽略它 → 高估压缩收益）
    assert prep.post_compact_estimate >= merged_tokens
    assert prep.post_compact_estimate <= 16384 - settings.reserve_tokens


def test_snapshot_cap_keeps_newest_entries():
    settings = CompactionSettings(
        memory_snapshot_max_tokens=128,
        keep_recent_tokens=4000,
        summary_max_tokens=1024,
    )
    prep = prepare_compaction(_region(), settings, context_window=16384)
    assert prep is not None
    memory = prep.context_state["memory"]
    # 裁剪后仍有记忆可用（不是整个注入块消失），且小于区域内的条目总数
    assert 0 < len(memory) < 41
    assert "<remembered-context>" in prep.merged_context
    assert "记忆" in prep.merged_context


async def test_compact_uses_summary_max_tokens(monkeypatch):
    import connectclaw.agent.harness.compaction as C
    from connectclaw.provider.types import Model

    async def fake_generate(*args, **kwargs):
        return "长" * 20000

    monkeypatch.setattr(C, "generate_summary", fake_generate)
    prep = C.CompactionPreparation(
        first_kept_entry_id="m1",
        messages_to_summarize=[{"role": "user", "content": "hi"}],
    )
    settings = CompactionSettings(summary_max_tokens=256)
    result = await C.compact(prep, Model(id="test-model"), settings=settings)
    assert count_tokens(result.summary) <= 320  # 预算 + 截断提示


async def test_generate_summary_forwards_max_tokens(monkeypatch):
    import connectclaw.agent.harness.compaction as C
    from connectclaw.provider.types import AssistantMessage, Model, StreamEvent

    captured: dict = {}

    def fake_stream(model, context, **kwargs):
        captured.update(kwargs)

        async def _gen():
            msg = AssistantMessage(
                content=[{"type": "text", "text": "summary"}],
                model="test-model", stop_reason="stop", timestamp=1.0,
            )
            yield StreamEvent(type="done", message=msg)

        return _gen()

    monkeypatch.setattr(C, "stream_simple", fake_stream)
    out = await C.generate_summary(
        [{"role": "user", "content": "hi"}], Model(id="test-model"), max_tokens=777)
    assert captured["max_tokens"] == 777
    assert out == "summary"


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


# ── 7. 摘要输入保真：按角色头尾截断 + thinking/toolCall（2026-10-22）──


def test_serialize_keeps_tool_result_head_and_tail():
    # 旧实现每条消息只留前 300 字符，工具输出的结论/报错常在后半段。
    body = "HEAD-" + ("x" * 20_000) + "-TAIL"
    out = _serialize([{"role": "toolResult", "content": body}])
    assert "HEAD-" in out
    assert "-TAIL" in out
    assert "chars omitted" in out


def test_serialize_includes_thinking_and_tool_calls():
    msgs = [{
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "先检查配置再启动服务"},
            {"type": "text", "text": "我来执行。"},
            {"type": "toolCall", "id": "c1", "name": "bash",
             "arguments": {"command": "systemctl restart demo"}},
        ],
    }]
    out = _serialize(msgs)
    assert "[thinking]" in out
    assert "先检查配置再启动服务" in out
    assert "[toolCall bash" in out
    assert "systemctl restart demo" in out


def test_extract_file_ops_from_tool_call_arguments():
    # 真实会话里 toolResult 不带 details；路径必须从配对的 toolCall 参数里取。
    msgs = [
        {"role": "assistant", "content": [
            {"type": "toolCall", "id": "r1", "name": "hash_read",
             "arguments": {"path": "/proj/a.py"}},
            {"type": "toolCall", "id": "e1", "name": "hash_edit",
             "arguments": {"path": "/proj/b.py"}},
            {"type": "toolCall", "id": "w1", "name": "write",
             "arguments": {"path": "/proj/c.py"}},
        ]},
        {"role": "toolResult", "tool_call_id": "r1", "tool_name": "hash_read"},
        {"role": "toolResult", "tool_call_id": "e1", "tool_name": "hash_edit"},
        {"role": "toolResult", "tool_call_id": "w1", "tool_name": "write"},
    ]
    ops = _extract_file_ops(msgs)
    assert ops.read == {"/proj/a.py"}
    assert ops.edited == {"/proj/b.py", "/proj/c.py"}


def test_refuses_compaction_that_cannot_shrink():
    # 摘要和快照把省下来的 token 又吃回去 → 不落盘，避免"压了等于没压"。
    entries = _mk("m", 40, "内容" * 40)
    settings = CompactionSettings(
        keep_recent_tokens=100, summary_max_tokens=100_000,
    )
    assert prepare_compaction(list(entries), settings, context_window=65536) is None
