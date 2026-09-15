"""Token estimation primitives — owned by the provider layer.

Token counting is a property of the model/vendor, not of any single consumer
(compaction, budget checks, serialization all need it), so it lives here.

Two categories:
  * text — tiktoken cl100k_base (DeepSeek-compatible) with a CJK-aware
    char heuristic as an offline fallback.
  * image — the API bills vision blocks at a fixed per-image cost (not the
    base64 length), so we estimate by byte-size tiers. Wide/fast enough.

The naive chars/4 heuristic under-counted CJK text by ~2-4x, which silently
blew keep-recent cuts over budget and caused a per-turn compaction livelock;
see module history in compaction.py.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import tiktoken

# ── Text counting ──────────────────────────────────────────────

_tokenizer_cache: dict[str, Any] = {}
_tokenizer_failure: Exception | None = None

# Memoized count_tokens results, bounded by total cached characters (not entry
# count — some tool results are multi-KB). Compaction rescans the same message
# set repeatedly across its cut-point iterations, so this turns O(iterations)
# tiktoken passes into cache hits.
_token_memo: OrderedDict[str, int] = OrderedDict()
_token_memo_chars = 0
_TOKEN_MEMO_MAX_CHARS = 4 * 1024 * 1024
_TOKEN_MEMO_MAX_ITEM = 64 * 1024


def _get_tokenizer(name: str = "cl100k_base") -> Any | None:
    """Cached tiktoken encoding; None on any failure (e.g. offline)."""
    global _tokenizer_failure
    if name in _tokenizer_cache:
        return _tokenizer_cache[name]
    if _tokenizer_failure is not None:
        return None
    try:
        enc = tiktoken.get_encoding(name)
        _tokenizer_cache[name] = enc
        return enc
    except Exception as exc:  # pragma: no cover - offline / download issue
        _tokenizer_failure = exc
        return None


def _is_cjk(ch: str) -> bool:
    # CJK ranges: Hiragana/Katakana, CJK ext A / Unified / Compatibility,
    # halfwidth katakana, CJK symbols. Single-char string compare == code point.
    return any(
        lo <= ch <= hi
        for lo, hi in (
            ("\u3040", "\u30ff"),
            ("\u3400", "\u4dbf"),
            ("\u4e00", "\u9fff"),
            ("\uf900", "\ufaff"),
            ("\uff66", "\uff9f"),
            ("\u3000", "\u303f"),
        )
    )


def count_tokens(text: str) -> int:
    """Best-effort text token count: tiktoken cl100k_base, else CJK-aware heuristic."""
    if not text:
        return 0
    cached = _token_memo.get(text)
    if cached is not None:
        _token_memo.move_to_end(text)
        return cached

    enc = _get_tokenizer()
    if enc is not None:
        try:
            result = max(1, len(enc.encode(text, disallowed_special=())))
        except Exception:
            enc = None
    if enc is None:
        cjk = sum(1 for ch in text if _is_cjk(ch))
        result = max(1, cjk + (len(text) - cjk) // 4)

    if len(text) <= _TOKEN_MEMO_MAX_ITEM:
        global _token_memo_chars
        while _token_memo and _token_memo_chars + len(text) > _TOKEN_MEMO_MAX_CHARS:
            _token_memo_chars -= len(_token_memo.popitem(last=False)[0])
        _token_memo[text] = result
        _token_memo_chars += len(text)
    return result


# ── Image counting ─────────────────────────────────────────────

_IMAGE_ESTIMATE_TIERS: list[tuple[int, int]] = [
    (64 * 1024, 400),      # tiny  (web favicon / small screenshots)
    (256 * 1024, 800),     # small (typical chat screenshots)
    (1024 * 1024, 1200),   # medium
]
_IMAGE_ESTIMATE_CAP = 1700
_IMAGE_ESTIMATE_MAX = 2048


def estimate_image_cost(size_bytes: int) -> int:
    """Fixed per-image token estimate by encoded byte size.

    Providers bill vision blocks at a fixed tile-based cost, never by base64
    length; byte-size tiers are the cheapest proxy we have without decoding
    the image (no PIL dependency). Bounded and never zero so budget math
    always has something to account for.
    """
    for limit, cost in _IMAGE_ESTIMATE_TIERS:
        if size_bytes <= limit:
            return cost
    return min(_IMAGE_ESTIMATE_MAX, _IMAGE_ESTIMATE_CAP + (size_bytes // (8 * 1024 * 1024)) * 300)


def image_block_cost(block: dict) -> int:
    """Token cost of an image / image_ref content block."""
    size = block.get("size")
    if size:
        return estimate_image_cost(int(size))
    # Legacy inline block: {'type': 'image', 'data': <base64>, ...}
    data = block.get("data")
    if isinstance(data, str) and data:
        return estimate_image_cost(int(len(data) * 3 / 4))  # base64 -> bytes
    return _IMAGE_ESTIMATE_TIERS[0][1]
