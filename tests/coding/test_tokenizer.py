"""Provider tokenizer tests.

Covers text counting (tiktoken real tokens, CJK not under-counted) and the
fixed per-image cost estimator used by compaction.
"""

from __future__ import annotations

from connectclaw.provider.tokenizer import (
    count_tokens,
    estimate_image_cost,
    image_block_cost,
)


# ── Text ───────────────────────────────────────────────────────


def test_count_tokens_nonempty_is_positive():
    assert count_tokens("") == 0
    assert count_tokens("hello world") >= 1
    assert count_tokens("中文字符串测试") >= 1


def test_count_tokens_cjk_not_undercounted():
    # chars/4 heuristic under-counted CJK; real tokenizer must not.
    sample = "中文" * 100
    assert count_tokens(sample) >= len(sample) // 4


def test_count_tokens_never_zero_for_nonempty():
    assert count_tokens(" ") >= 1
    assert count_tokens("\n\n") >= 1


# ── Image cost tiers ───────────────────────────────────────────


def test_estimate_image_cost_tiers():
    tiny = estimate_image_cost(10 * 1024)
    assert tiny == 400
    small = estimate_image_cost(100 * 1024)
    assert small == 800
    medium = estimate_image_cost(500 * 1024)
    assert medium == 1200
    large = estimate_image_cost(5 * 1024 * 1024)
    assert large >= 1200


def test_estimate_image_cost_bounded_and_positive():
    # Huge / degenerate inputs stay bounded and never return zero.
    assert estimate_image_cost(0) == 400
    assert estimate_image_cost(10 ** 9) <= 2048
    assert estimate_image_cost(10 ** 9) >= 1


def test_image_block_cost_variants():
    # image_ref by size; legacy inline 'image' by base64 length.
    assert image_block_cost({"type": "image_ref", "size": 100 * 1024}) == 800
    assert image_block_cost({"type": "image", "data": "AAAA"}) >= 1
    # No size / data → smallest tier (never zero).
    assert image_block_cost({"type": "image_ref"}) == 400
