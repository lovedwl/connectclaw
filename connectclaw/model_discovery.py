"""Automatic model discovery for /model.

Fetches the OpenAI-compatible ``{base}/models`` endpoint with a provider's
key, parses per-model limits (``max_input_tokens`` / ``max_output_tokens``
when the gateway exposes them), and — for models whose limit is not exposed —
can probe the real context window by sending an intentionally oversized
prompt and reading the limit out of the gateway's 400 error (litellm /
OpenAI-style gateways echo "maximum context length is N tokens" without
billing: the request is rejected before inference).

Discovery is read-only: nothing here writes the registry. Saving is done by
the callers (model_card stages / the /model sync wizard) via ModelsStore.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from connectclaw.logging import get_logger
from connectclaw.model_registry import ModelProfile, normalize_url

logger = get_logger(__name__)

DEFAULT_CONTEXT_WINDOW = 65536
DEFAULT_MAX_TOKENS = 8192

# Oversized-prompt probes. Filler must be token-dense: a repeated word like
# "probe " compresses to very few BPE tokens, so the gateway never sees an
# oversized prompt and the 400 never fires. Cycling a small word list with
# a coprime stride keeps ~2 tokens/word with a modest body: small ≈ 160k
# tokens (~0.8MB) resolves windows ≤128k; large ≈ 640k tokens (~2.2MB)
# resolves up to ~500k. Both are rejected pre-inference by well-behaved
# gateways, so a probe costs (almost) nothing.
_PROBE_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu".split()
)


def _probe_content(n_words: int) -> str:
    ws = _PROBE_WORDS
    # stride 7 is coprime with len(ws) → no adjacent repeats, uniform cycle
    return " ".join(ws[(i * 7) % len(ws)] for i in range(n_words))


_PROBE_SMALL = _probe_content(80_000)
_PROBE_LARGE = _probe_content(320_000)
_PROBE_TIMEOUT = 60.0

# Ordered — first hit wins. Covers the gateway/litellm wording observed in the
# wild, OpenAI's, and Anthropic's "X tokens > Y maximum" form.
_LIMIT_PATTERNS = [
    re.compile(r"maximum context length is (\d[\d,]*)", re.I),
    re.compile(r"context length is (\d[\d,]*)", re.I),
    re.compile(r"max(?:imum)?[_ ]?(?:context[_ ]?)?tokens? is (\d[\d,]*)", re.I),
    re.compile(r"(\d[\d,]*)\s*tokens?\s*[>＞]\s*(\d[\d,]*)", re.I),  # take the cap (group 2)
    re.compile(r"max_?input_?tokens[\"'=:\s]+(\d[\d,]*)", re.I),
]


class DiscoveryError(Exception):
    """Readable failure for the /models fetch (surfaced in card/chat)."""


class AuthError(DiscoveryError):
    """key 被拒（401/403）——通常意味着 key 已被封禁/吊销；
    同 key 的全部 profile 均不可用，卡片会转入清理模式。"""

def _int_field(m: dict, *keys: str) -> int | None:
    for k in keys:
        v = m.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
        if isinstance(v, str) and v.strip().lstrip("-").isdigit():
            n = int(v.strip())
            if n > 0:
                return n
    return None


def parse_entry(m: dict) -> dict:
    """One /models list item -> {id, max_input, max_output} (None when absent)."""
    return {
        "id": str(m["id"]),
        "max_input": _int_field(m, "max_input_tokens", "max_input",
                                "context_length", "max_context", "max_input_len"),
        "max_output": _int_field(m, "max_output_tokens", "max_output"),
    }


def fmt_ctx(n: int | None) -> str:
    """262144 -> '256k', 1048576 -> '1M', None -> '?'."""
    if not n or n <= 0:
        return "?"
    if n >= 1_024_000:
        m = f"{n / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{m}M"
    if n >= 1024:
        return f"{round(n / 1024)}k"
    return str(n)


def _host(url: str) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower()
    except ValueError:
        return ""


def match_existing(profiles: list[ModelProfile], base_url: str, model_id: str) -> ModelProfile | None:
    """Profile with the same model_id on the same host (ignoring /v1 suffix
    and path differences in the stored base_url)."""
    h = _host(base_url)
    if not h:
        return None
    for p in profiles:
        if p.model_id == model_id and _host(p.base_url) == h:
            return p
    return None


async def fetch_model_list(
    base_url: str,
    api_key: str = "",
    proxy: str | None = None,
    timeout: float = 12.0,
) -> tuple[list[dict], str]:
    """GET {base}/models (falling back to {base}/v1/models like /model verify).

    Returns (parsed entries, endpoint actually used). Raises DiscoveryError
    with a user-readable message on any failure.
    """
    base = (normalize_url(base_url) or "").strip().rstrip("/")
    if "://" not in base:
        raise DiscoveryError(f"无效的 base_url：{base_url!r}")
    bases = [base] if base.endswith("/v1") else [base, base + "/v1"]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_err = ""
    async with httpx.AsyncClient(timeout=timeout, proxy=proxy or None, trust_env=False) as c:
        for i, cand in enumerate(bases):
            url = f"{cand}/models"
            try:
                r = await c.get(url, headers=headers)
            except Exception as e:  # noqa: BLE001 — network errors → readable
                last_err = f"{url} 无法访问：{e}"
                logger.warning("discovery fetch failed %s: %s", url, e)
                continue
            if r.status_code == 404 and i < len(bases) - 1:
                continue  # bare endpoint 404s → the /v1 prefix is next
            if r.status_code in (401, 403):
                raise AuthError(f"`{url}` → HTTP {r.status_code}（key 已失效或被封禁）")
            if r.status_code != 200:
                raise DiscoveryError(f"`{url}` → HTTP {r.status_code}（可能不是 OpenAI 兼容端点）")
            try:
                payload = r.json()
            except Exception:  # noqa: BLE001
                payload = {}
            entries = [parse_entry(m) for m in (payload.get("data") or [])
                       if isinstance(m, dict) and m.get("id")]
            return entries, cand
    raise DiscoveryError(last_err or f"{base}/models 无有效响应")


def _extract_limit(text: str) -> int | None:
    for pat in _LIMIT_PATTERNS:
        m = pat.search(text or "")
        if m:
            num = m.group(2) if pat is _LIMIT_PATTERNS[3] else m.group(1)
            try:
                n = int(num.replace(",", ""))
                if n > 0:
                    return n
            except ValueError:
                continue
    return None


async def probe_context_window(
    base_url: str,
    api_key: str,
    model_id: str,
    proxy: str | None = None,
    timeout: float = _PROBE_TIMEOUT,
) -> tuple[int | None, str]:
    """Probe the real context window via an oversized-prompt 400.

    Two escalating probe sizes; a 400 whose message names the limit wins.
    Returns (limit or None, human-readable note). Never retries past the
    large probe — a 200 there means the window is bigger than we probe for
    (and paying for real inference on a 1M+ token prompt to learn the exact
    number is not worth it).
    """
    base = (normalize_url(base_url) or "").strip().rstrip("/")
    if "://" not in base:
        return None, f"无效的 base_url：{base_url!r}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"{base}/chat/completions"
    last_err = ""
    saw_200 = False
    async with httpx.AsyncClient(timeout=timeout, proxy=proxy or None, trust_env=False) as c:
        for label, content in (("small", _PROBE_SMALL), ("large", _PROBE_LARGE)):
            body = {
                "model": model_id,
                "max_tokens": 1,
                "temperature": 0,
                "messages": [{"role": "user", "content": content}],
            }
            try:
                r = await c.post(url, json=body, headers=headers)
            except Exception as e:  # noqa: BLE001
                reason = str(e) or e.__class__.__name__
                return None, f"探测请求失败（{label} 级）：{reason}"
            if r.status_code == 200:
                saw_200 = True
                continue  # window bigger than this probe → escalate
            text = r.text or ""
            limit = _extract_limit(text)
            if limit:
                return limit, f"HTTP {r.status_code}（{label} 级探测命中）"
            if r.status_code in (401, 403, 404, 429):
                snippet = text[:140].replace("\n", " ")
                return None, f"HTTP {r.status_code}（非上下文错误，停止探测）：{snippet}"
            last_err = f"HTTP {r.status_code}：{text[:140].replace(chr(10), ' ')}"
            logger.warning("context probe (%s) %s → %s", label, url, last_err)
    if saw_200:
        return None, f"两级探测均返回 200 —— 窗口大于探测上限（约 1M+），请人工确认{'' if not last_err else '；另：' + last_err}"
    return None, (f"未能解析出上限 {last_err}" if last_err else "探测无有效响应")
