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
    _serialize,
    count_tokens,
    estimate_tokens,
    find_cut_point,
    prepare_compaction,
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