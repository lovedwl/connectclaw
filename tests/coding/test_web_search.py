"""web_search RSS fast path + web_fetch page cache / prompt extraction.

Offline unit tests only: parse/filter/render/cache are pure functions; the LLM
extraction runs against a fake AsyncOpenAI. Browser orchestration needs real
network and is not tested (same philosophy as test_web_fetch.py).
"""

from __future__ import annotations

import asyncio
import types

import pytest

from connectclaw.coding.tools import lightpanda
from connectclaw.coding.tools.lightpanda import (
    _cache_get,
    _cache_put,
    _domain_allowed,
    _parse_bing_rss,
    _render_results,
)
from connectclaw.coding.tools.web_search import (
    WebFetchTool,
    WebSearchConfig,
    WebSearchTool,
)

# Fixture modeled on the real cn.bing.com RSS response (verified live).
_RSS_SAMPLE = """<?xml version="1.0" encoding="utf-8" ?>
<rss version="2.0"><channel>
<title>bing: python asyncio</title><description>search results</description>
<item><title>Python Docs</title><link>https://docs.python.org/</link>
<description>Official &lt;b&gt;asyncio&lt;/b&gt;  docs.</description></item>
<item><title>Real Python</title><link>https://realpython.com/asyncio/</link>
<description>Tutorials &amp; guides.</description></item>
<item><broken>no title/link here</broken></item>
</channel></rss>"""


# ── RSS parsing / filtering / rendering ──────────────────────────


def test_parse_bing_rss():
    results = _parse_bing_rss(_RSS_SAMPLE)
    assert len(results) == 2
    assert results[0]["title"] == "Python Docs"
    assert results[0]["url"] == "https://docs.python.org/"
    # tags stripped, entities decoded, whitespace squeezed
    assert results[0]["snippet"] == "Official asyncio docs."


def test_parse_bing_rss_garbage_returns_empty():
    assert _parse_bing_rss("<html><body>blocked</body></html>") == []
    assert _parse_bing_rss("not xml at all") == []


def test_domain_allowed():
    assert _domain_allowed("https://docs.python.org/3/", ["python.org"], None)
    assert _domain_allowed("https://example.com/", None, None)
    assert not _domain_allowed("https://docs.python.org/", ["pypi.org"], None)
    # blocked blocks subdomains too
    assert not _domain_allowed("https://mail.spam.com/", None, ["spam.com"])
    # allowed is exact-host or dot-suffix — not a bare substring
    assert not _domain_allowed("https://notpython.org/", ["python.org"], None)


def test_render_results():
    out = _render_results([
        {"title": "A", "url": "https://a.com/", "snippet": "first"},
        {"title": "B", "url": "https://b.com/", "snippet": ""},
    ])
    assert "1. [A](https://a.com/)" in out
    assert "   first" in out
    assert "2. [B](https://b.com/)" in out


# ── page cache ───────────────────────────────────────────────────


class _FakeTime:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


@pytest.fixture()
def fake_time(monkeypatch):
    ft = _FakeTime()
    monkeypatch.setattr(lightpanda, "time", ft)
    monkeypatch.setattr(lightpanda, "_fetch_cache", {})
    return ft


def test_cache_hit_and_expiry(fake_time):
    _cache_put("https://a.com", "PAGE")
    assert _cache_get("https://a.com") == "PAGE"
    fake_time.now += 901  # past the 900s TTL
    assert _cache_get("https://a.com") is None


def test_cache_eviction_caps_entries(fake_time):
    for i in range(lightpanda._FETCH_CACHE_MAX + 1):
        _cache_put(f"https://x.com/{i}", f"page {i}")
    assert _cache_get("https://x.com/0") is None  # oldest evicted
    assert _cache_get(f"https://x.com/{lightpanda._FETCH_CACHE_MAX}") is not None


async def test_fetch_once_uses_cache(fake_time, monkeypatch):
    calls: list[str] = []

    async def fake_http(url, timeout=0.0):
        calls.append(url)
        return "content " * 100, ""      # (text, reason) —— 直连成功时 reason 为空

    monkeypatch.setattr(lightpanda, "http_fetch_once", fake_http)
    text1 = await lightpanda.fetch_once("https://a.com/page", 8000)
    text2 = await lightpanda.fetch_once("https://a.com/page", 8000)
    assert calls == ["https://a.com/page"]  # second call served from cache
    assert text1 == text2
    await lightpanda.fetch_once("https://a.com/page", 8000, no_cache=True)
    assert len(calls) == 2  # no_cache bypasses the cache


# ── tool layer ───────────────────────────────────────────────────


@pytest.fixture()
def fetch_page(monkeypatch):
    async def fake_fetch(url, max_chars=8000, http_timeout=0.0, no_cache=False):
        return "PAGE CONTENT " * 50

    monkeypatch.setattr(lightpanda, "fetch_once", fake_fetch)


async def test_web_fetch_plain(fetch_page):
    tool = WebFetchTool(WebSearchConfig())
    result = await tool.execute("tc", {"url": "https://a.com"})
    assert "PAGE CONTENT" in result.content[0]["text"]
    assert result.details["url"] == "https://a.com"


class _FakeOpenAI:
    """Stand-in for openai.AsyncOpenAI recording the create() kwargs."""

    last_create: dict | None = None

    def __init__(self, **kwargs):
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        _FakeOpenAI.last_create = kwargs
        msg = types.SimpleNamespace(content="The answer is 42.")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


async def test_web_fetch_prompt_extraction(fetch_page, monkeypatch):
    import openai

    _FakeOpenAI.last_create = None
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeOpenAI)

    model = types.SimpleNamespace(
        id="main-model", base_url="https://api.example.com", proxy=""
    )
    tool = WebFetchTool(
        WebSearchConfig(),
        model_provider=lambda: model,
        api_key_provider=lambda: "sk-test",
    )
    result = await tool.execute("tc", {"url": "https://a.com", "prompt": "life?"})
    assert result.content[0]["text"] == "The answer is 42."
    assert result.details["extracted"] is True
    kw = _FakeOpenAI.last_create
    assert kw["model"] == "main-model"
    # the page content and the question both reach the extraction model
    assert "life?" in kw["messages"][1]["content"]
    assert "PAGE CONTENT" in kw["messages"][1]["content"]


async def test_web_fetch_prompt_degrades_without_model(fetch_page):
    tool = WebFetchTool(WebSearchConfig())  # no model provider → extraction unavailable
    result = await tool.execute("tc", {"url": "https://a.com", "prompt": "life?"})
    text = result.content[0]["text"]
    assert "PAGE CONTENT" in text
    assert "prompt 提取不可用" in text
    assert result.details["extracted"] is False


async def test_web_search_tool_passes_params(monkeypatch):
    captured: dict = {}

    async def fake_search(query, max_chars=8000, **kwargs):
        captured.update(kwargs)
        captured["query"] = query
        captured["max_chars"] = max_chars
        return "1. [A](https://a.com/)\n   snippet"

    monkeypatch.setattr(lightpanda, "search_once", fake_search)
    tool = WebSearchTool(WebSearchConfig(timeout=17))
    result = await tool.execute("tc", {
        "query": "q",
        "max_results": 5,
        "allowed_domains": ["a.com"],
    })
    assert captured["query"] == "q"
    assert captured["max_results"] == 5
    assert captured["allowed_domains"] == ["a.com"]
    assert captured["timeout"] == 17.0
    assert "1. [A](https://a.com/)" in result.content[0]["text"]


# ── fail open：真实错误要暴露给 agent ────────────────────────
#
# 2026-09-25 用户实测：web_fetch huggingface.co 只报"浏览器引擎在此页面失败
# （Lightpanda 仍为 Beta；繁重或 JS 框架页面可能使其崩溃）"——把"外网不可达"
# 错报成"引擎太脆"，agent 因此不知道要换路（走代理是它凭记忆该做的判断）。
# 修法：不揣测原因、不建议走代理，**原样抛出真实错误**；并且直连 GET 的报错
# 不能再被静默吞掉（那条连接层错误最有诊断价值）。


async def test_fetch_failure_exposes_both_real_errors(monkeypatch):
    async def fake_http(url, timeout=0.0):
        return "", "ConnectTimeout: "

    async def fake_run(action, nav_timeout=0):
        raise lightpanda.LightpandaError(
            "浏览器会话失败：ConnectionClosedError: no close frame received or sent"
        )

    monkeypatch.setattr(lightpanda, "http_fetch_once", fake_http)
    monkeypatch.setattr(lightpanda, "_run_stateless", fake_run)

    with pytest.raises(lightpanda.LightpandaError) as exc:
        await lightpanda.fetch_once("https://huggingface.co/convaiinnovations", 5000, no_cache=True)

    msg = str(exc.value)
    assert "ConnectTimeout" in msg, "直连的真实报错必须出现（过去被静默吞掉）"
    assert "no close frame received or sent" in msg, "引擎的真实报错也要在"
    assert "Beta" not in msg, "不要再把原因揣测成'引擎太脆'"
    assert "代理" not in msg and "镜像" not in msg, "工具不该给'思路'，那是记忆的事"


async def test_http_fetch_once_reports_connect_error(monkeypatch):
    """连接层异常要作为 reason 返回，而不是空串。"""
    import httpx

    class _Boom:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("[Errno 111] Connection refused")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(lightpanda.httpx, "AsyncClient", _Boom)
    text, reason = await lightpanda.http_fetch_once("https://example.invalid/x")
    assert text == ""
    assert "ConnectError" in reason and "Connection refused" in reason


async def test_run_stateless_raises_raw_error(monkeypatch):
    """浏览器会话失败时，抛的是原始异常，不再套"Lightpanda 仍为 Beta"的帽子。"""
    class _Eng:
        def __init__(self, *a, **kw):
            pass

        async def attach(self, url):
            raise ConnectionResetError("[Errno 104] Connection reset by peer")

    async def fake_server():
        return "ws://127.0.0.1:9222/x"

    monkeypatch.setattr(lightpanda, "LightpandaEngine", _Eng)
    monkeypatch.setattr(lightpanda, "_ensure_server", fake_server)
    lightpanda._sem = asyncio.Semaphore(1)

    with pytest.raises(lightpanda.LightpandaError) as exc:
        await lightpanda._run_stateless(lambda eng, sid: None)
    assert "ConnectionResetError" in str(exc.value)
    assert "Connection reset by peer" in str(exc.value)
    assert "Beta" not in str(exc.value)
