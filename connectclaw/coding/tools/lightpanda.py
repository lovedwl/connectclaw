"""
Lightpanda browser engine — CDP over websocket, no pixel rendering.

Powers both:
  - web_search / web_fetch  (stateless: open a fresh page per call, shared engine)
  - the `browser` stateful tool script (its own engine + a kept page/session)

Verified CDP flow (lightpanda-py 0.2.8):
  serve → GET /json/version → ws → Target.createTarget → Target.attachToTarget
  → (with sessionId) Page.enable / Runtime.enable / Page.navigate / Runtime.evaluate

Lightpanda is DOM-focused: **no screenshots**, use readyState polling not
networkidle, reuse ONE ws connection and open pages (targets) on it.

Default network policy: bare subprocess with full network, NOT sandboxed
(browser must reach the internet; the sandbox is for untrusted shell anyway).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import html.parser
import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

import httpx
import websockets

from connectclaw.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PORT = 9222
DEFAULT_NAV_TIMEOUT = 20          # s, page navigation budget
_SERVE_INACTIVITY = 3600          # s, keep CDP alive for long-lived sessions
_RECV_SLACK = 15                  # s added to nav timeout for ws recv

# ── 内存边界（2026-09-25 事故后加的）─────────────────────────
# 事故：一次 web_search 之后 unit 峰值 **15.5G**，被内核 OOM 杀掉，整机跟着 swap 抖动
# （那次运行里 4 次 Lightpanda 抓取全部失败："no close frame received or sent"）。
# 这个文件里有三处**无界**缓冲，任何一处撞上繁重页面都能吃满内存：
#   1) CDP websocket 原本 `max_size=None` —— 单条消息没有上限，整页 DOM/文本一次性
#      读进 Python 进程；
#   2) 页面文本**全量**进 LRU 缓存，只按条数（32）封顶，不按大小；
#   3) 共享的 `lightpanda serve` 子进程（Beta 引擎）持有页面，既没上限也没观测
#      —— 代码里已经记着一次"orphan lightpanda 堆到 21G 峰值"的前科。
# 下面把三处都钉上界，并在每次浏览器会话后观测子进程 RSS，超硬阈值就重启它
# （把 15G 猝死变成 3G 重启），下次再犯也能直接定位。
_MAX_PAGE_CHARS = 2_000_000                  # 单页文本上限（正常页面可见文本 <200KB）
_CDP_MAX_MESSAGE_BYTES = 32 * 1024 * 1024    # 单条 CDP 消息上限
_FETCH_CACHE_MAX_CHARS = 8_000_000           # web_fetch 缓存总量上限
_CHILD_RSS_WARN_MB = 1024                    # 子进程 RSS 超此值告警
_CHILD_RSS_KILL_MB = 2048                    # 超此值重启子进程（防 OOM）


class LightpandaError(RuntimeError):
    pass


# JS payloads (kept tiny; Lightpanda runs V8 but not the full web platform).
_EXTRACT_JS = (
    "(function(){var t=document.title||'';"
    "var b=document.body?document.body.innerText:'';"
    "return JSON.stringify({title:t,text:b});})()"
)


def _click_js(selector: str) -> str:
    s = json.dumps(selector)
    return (
        f"(function(){{var el=document.querySelector({s});"
        f"if(!el)return false;el.click();return true;}})()"
    )


def _type_js(selector: str, text: str) -> str:
    s, t = json.dumps(selector), json.dumps(text)
    return (
        f"(function(){{var el=document.querySelector({s});if(!el)return false;"
        f"el.focus();el.value={t};"
        f"el.dispatchEvent(new Event('input',{{bubbles:true}}));"
        f"el.dispatchEvent(new Event('change',{{bubbles:true}}));return true;}})()"
    )


def _serve_process(port: int, *, cdp_max_connections: int | None = None):
    """Start a Lightpanda CDP server subprocess (logs discarded).

    cdp_max_connections lifts the server-side cap on simultaneous CDP
    connections (default 16) for the shared multi-client server.
    """
    import lightpanda  # lightpanda-py bundles the binary

    devnull = open(os.devnull, "w")
    # lightpanda.serve() also prints a banner to the caller's stdout; swallow it
    # so it can never pollute a stateful tool's stdout line protocol.
    with contextlib.redirect_stdout(devnull):
        return lightpanda.serve(
            host="127.0.0.1",
            port=port,
            timeout=_SERVE_INACTIVITY,
            log_level="error",
            cdp_max_connections=cdp_max_connections,
            stdout=devnull,
            stderr=devnull,
        )


class LightpandaEngine:
    """One Lightpanda serve process + one persistent ws connection.

    Open multiple pages (CDP targets) on the single connection. RPC is
    serialized by a lock (one in-flight request per connection); CDP events
    (messages without our id) are skipped.
    """

    def __init__(self, *, port: int = DEFAULT_PORT, nav_timeout: int = DEFAULT_NAV_TIMEOUT):
        self._port = port
        self._nav_timeout = nav_timeout
        self._proc: Any = None
        self._ws: Any = None
        self._mid = 0
        self._lock = asyncio.Lock()
        self._sid_to_tid: dict[str, str] = {}

    # ── lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Own a fresh serve process + a ws connection to it (stateful use)."""
        if self._ws is not None:
            return
        self._proc = _serve_process(self._port)
        # Register the kill hook the instant we own a child process — BEFORE the
        # readiness probe, which can raise. If _await_ready() times out (e.g. the
        # port is already taken), the child is still alive; without early
        # registration + the explicit kill below it would leak as an orphan
        # `lightpanda serve` and pile up across retries (the 21G-peak bug).
        atexit.register(self._sync_kill)
        try:
            ws_url = await self._await_ready()
            self._ws = await websockets.connect(
            ws_url, max_size=_CDP_MAX_MESSAGE_BYTES, open_timeout=15
        )
        except BaseException:
            self._sync_kill()
            raise

    async def attach(self, ws_url: str) -> None:
        """Attach to an ALREADY-RUNNING serve process (shared, stateless use).

        Opens a *new* ws connection to an existing server: one serve process can
        host many CDP connections (cdp_max_connections defaults to 16), and each
        connection carries its own target. This is the multi-client model — N
        concurrent browser sessions on ONE process, no per-engine subprocess, no
        port juggling. This engine does NOT own the process, so close() only
        drops the ws; the server outlives it.
        """
        if self._ws is not None:
            return
        self._proc = None  # not ours — never kill it
        self._ws = await websockets.connect(
            ws_url, max_size=_CDP_MAX_MESSAGE_BYTES, open_timeout=15
        )

    async def _await_ready(self) -> str:
        url = f"http://127.0.0.1:{self._port}/json/version"
        async with httpx.AsyncClient() as c:
            for _ in range(120):
                try:
                    r = await c.get(url, timeout=1)
                    return r.json()["webSocketDebuggerUrl"]
                except Exception:
                    await asyncio.sleep(0.15)
        raise LightpandaError("Lightpanda CDP 服务未能及时就绪")

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._sync_kill()

    def _sync_kill(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    # ── rpc ──────────────────────────────────────────────────

    async def _rpc(self, method: str, params: dict | None = None, sid: str | None = None) -> dict:
        if self._ws is None:
            raise LightpandaError("引擎未启动")
        async with self._lock:
            self._mid += 1
            mid = self._mid
            msg: dict[str, Any] = {"id": mid, "method": method, "params": params or {}}
            if sid:
                msg["sessionId"] = sid
            await self._ws.send(json.dumps(msg))
            while True:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=self._nav_timeout + _RECV_SLACK)
                m = json.loads(raw)
                if m.get("id") == mid:
                    if "error" in m:
                        raise LightpandaError(f"{method}: {m['error']}")
                    return m.get("result", {})
                # otherwise a CDP event — ignore

    # ── pages ────────────────────────────────────────────────

    async def open_page(self) -> str:
        r = await self._rpc("Target.createTarget", {"url": "about:blank"})
        tid = r["targetId"]
        a = await self._rpc("Target.attachToTarget", {"targetId": tid, "flatten": True})
        sid = a["sessionId"]
        self._sid_to_tid[sid] = tid
        await self._rpc("Page.enable", sid=sid)
        await self._rpc("Runtime.enable", sid=sid)
        return sid

    async def close_page(self, sid: str) -> None:
        tid = self._sid_to_tid.pop(sid, None)
        if tid:
            try:
                await self._rpc("Target.closeTarget", {"targetId": tid})
            except Exception:
                pass

    # ── ops ──────────────────────────────────────────────────

    async def _eval(self, sid: str, expression: str) -> Any:
        r = await self._rpc(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
            sid=sid,
        )
        return (r.get("result") or {}).get("value")

    async def navigate(self, sid: str, url: str) -> None:
        await self._rpc("Page.navigate", {"url": url}, sid=sid)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._nav_timeout
        while loop.time() < deadline:
            state = await self._eval(sid, "document.readyState")
            if state in ("interactive", "complete"):
                return
            await asyncio.sleep(0.2)

    async def read_markdown(self, sid: str) -> str:
        """Page content as Markdown.

        Prefers Lightpanda's native `LP.getMarkdown` CDP command — it walks the
        rendered DOM into structured Markdown (headings, links, list items), so
        e.g. Bing results come back as `## [title](url)\\n snippet` instead of one
        flattened innerText blob. Falls back to raw innerText if the command is
        unavailable (older engine) or errors on a page.
        """
        try:
            r = await self._rpc("LP.getMarkdown", {}, sid=sid)
            md = r.get("markdown")
            if isinstance(md, str) and md.strip():
                # Already structured Markdown — keep its newlines (collapse only
                # runs of 3+ blank lines), don't flatten like innerText. Strip
                # image syntax first (token bloat + breaks Feishu cards).
                return _cap_page(_squeeze_blanklines(_strip_images(md).strip()))
        except LightpandaError:
            pass  # command missing / page too heavy — fall back below

        raw = await self._eval(sid, _EXTRACT_JS)
        try:
            data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (json.JSONDecodeError, TypeError):
            data = {"title": "", "text": str(raw or "")}
        title = (data.get("title") or "").strip()
        text = _collapse(data.get("text") or "")
        return _cap_page(f"# {title}\n\n{text}" if title else text)

    async def click(self, sid: str, selector: str) -> bool:
        return bool(await self._eval(sid, _click_js(selector)))

    async def type(self, sid: str, selector: str, text: str) -> bool:
        return bool(await self._eval(sid, _type_js(selector, text)))


# ── text helpers ────────────────────────────────────────────


def _collapse(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").replace("\r", "").split("\n")]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if ln.strip():
            blank = 0
            out.append(ln)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


# Markdown image syntax: ![alt](url). LP.getMarkdown emits these for every page
# image — tracking pixels, avatars, and huge base64 data: URIs. They are useless
# to a text agent AND poison Feishu cards: the card renderer treats `![](url)` as
# an image element needing a valid Feishu image_key, so an external/bing URL
# fails the whole card patch with "card contains invalid image keys". Strip them.
_IMG_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Empty-text links `[](url)` are nav-icon artifacts — drop the link, they render
# as bare "()" noise. Keep links that have visible text.
_EMPTY_LINK_RE = re.compile(r"\[\s*\]\([^)]*\)")


def _strip_images(text: str, *, cleanup_parens: bool = True) -> str:
    text = _IMG_MD_RE.sub("", text)
    text = _EMPTY_LINK_RE.sub("", text)
    if cleanup_parens:
        # Tidy up the "()" leftovers a removed inline element can leave behind.
        # Only for the LP.getMarkdown path — the HTML→Markdown converter emits
        # no link remnants, and blanking "()" would eat inline code like
        # `foo()`.
        text = re.sub(r"\(\s*\)", "", text)
    return text


# Bing chrome markers. The real results sit between "约 N 个结果 / N results" and
# the legal disclaimer / pager / footer. Trimming to that window drops the header
# nav (国内版/图片/视频…) and the footer noise, leaving just the result list.
_BING_HEAD_RE = re.compile(r"约\s*[\d,，]+\s*个?\s*结果|[\d,]+\s+[Rr]esults")
# The footer starts at the legal disclaimer ("为回应符合本地法律…") which Bing
# appends right after the last result; the pager and ICP filing follow it.
_BING_TAIL_RE = re.compile(
    r"为回应符合本地法律|部分搜索结果未予显示|分页\s*\d|相关搜索|Related searches|"
    r"©\s*\d{4}\s*Microsoft|增值电信业务|隐私条款"
)


def _trim_bing_chrome(text: str) -> str:
    """Cut Bing's header nav and footer, keeping just the result list."""
    m = _BING_HEAD_RE.search(text)
    if m:
        # Keep from just after the "约 N 个结果" marker (skip its line).
        nl = text.find("\n", m.end())
        text = text[nl + 1:] if nl != -1 else text[m.end():]
    t = _BING_TAIL_RE.search(text)
    if t:
        text = text[: t.start()]
        # The disclaimer often follows the last result's serial number ("11."),
        # leaving a dangling "11." — strip a trailing orphan number.
        text = re.sub(r"\n\s*\d+\.\s*$", "", text).rstrip()
    return text.strip()


def _squeeze_blanklines(text: str) -> str:
    """Collapse runs of 3+ blank lines to one, but keep single blanks and
    indentation — structured Markdown (headings, lists, links) must survive."""
    out: list[str] = []
    blank = 0
    for ln in text.replace("\r", "").split("\n"):
        if ln.strip():
            blank = 0
            out.append(ln.rstrip())
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


def _cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    note = f"\n…[已截断至 {max_chars} 字符]"
    return text[: max_chars - len(note)] + note


# ── shared stateless engine (web_search / web_fetch) ────────
#
# Multi-client model: ONE shared `serve` process hosts many concurrent CDP
# connections (cdp_max_connections defaults to 16). Each stateless call opens
# its OWN short-lived ws connection + target on that single process, so N calls
# run truly in parallel with NO per-call subprocess and NO port juggling.
#
# This replaces the old per-engine-subprocess pool, whose engines all reused the
# hardcoded port 9222 → only one could bind, the rest leaked orphan `serve`
# children on every retry (the 21G-peak bug) while real concurrency collapsed to
# 1. Here there is exactly one process to track and kill.
#
# The old note "concurrent createTarget → TargetAlreadyLoaded, needs multiple
# instances" was really about concurrent createTarget on ONE ws connection (RPC
# is serialized by a per-connection lock). Separate ws connections sidestep it.

_MAX_CONCURRENCY = 16          # simultaneous browser sessions (targets)
_CDP_MAX_CONNECTIONS = 16      # must be ≥ _MAX_CONCURRENCY (server-side cap)

_shared_proc: Any = None
_shared_ws_url: str | None = None
_shared_lock = asyncio.Lock()          # guards one-time server startup
_sem: asyncio.Semaphore | None = None  # caps concurrent sessions


def configure_pool(size: int) -> None:
    """Set max concurrency before first use (no-op once the server is up).

    Kept named `configure_pool` for its existing call site; `size` now means the
    max number of simultaneous browser sessions on the shared process.
    """
    global _MAX_CONCURRENCY, _CDP_MAX_CONNECTIONS
    if _shared_proc is None and size > 0:
        _MAX_CONCURRENCY = size
        _CDP_MAX_CONNECTIONS = max(_CDP_MAX_CONNECTIONS, size)


def _cap_page(text: str) -> str:
    """单页文本封顶。正常页面可见文本 <200KB，超了就是异常页面——
    别让它灌进 CDP 缓冲区、缓存和后续的 markdown 处理。"""
    if len(text) <= _MAX_PAGE_CHARS:
        return text
    logger.warning("lightpanda: 页面文本 %d 字超上限，截断至 %d 字", len(text), _MAX_PAGE_CHARS)
    return text[:_MAX_PAGE_CHARS]


def _child_rss_mb() -> float:
    """共享 lightpanda serve 子进程的当前 RSS（MB）；拿不到就返回 0。"""
    proc = _shared_proc
    if proc is None or proc.poll() is not None:
        return 0.0
    try:
        with open(f"/proc/{proc.pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _check_child_memory() -> None:
    """观测、并兜底共享浏览器子进程的内存。

    这是唯一能直接区分"内存是浏览器吃掉的还是 Python 吃掉的"的地方——15.5G 那次事故
    只能从 unit 峰值反推。超过硬阈值就重启子进程：把 15G 猝死降级成 3G 重启，
    不会再把整机拖进 swap（内核 OOM 杀 unit、桌面跟着卡死就是这么来的）。
    """
    rss = _child_rss_mb()
    if rss <= 0:
        return
    if rss >= _CHILD_RSS_KILL_MB:
        logger.warning(
            "lightpanda 子进程 RSS %.0f MB ≥ %.0f MB，重启它以防 OOM", rss, _CHILD_RSS_KILL_MB
        )
        _kill_shared_proc()
    elif rss >= _CHILD_RSS_WARN_MB:
        logger.warning("lightpanda 子进程 RSS 已达 %.0f MB（告警阈值 %.0f MB）", rss, _CHILD_RSS_WARN_MB)
    else:
        logger.debug("lightpanda 子进程 RSS %.0f MB", rss)


def _kill_shared_proc() -> None:
    global _shared_proc, _shared_ws_url
    proc, _shared_proc, _shared_ws_url = _shared_proc, None, None
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass


async def _ensure_server() -> str:
    """Start the single shared serve process (once) and return its ws url."""
    global _shared_proc, _shared_ws_url, _sem
    if _shared_ws_url is not None:
        return _shared_ws_url
    async with _shared_lock:
        if _shared_ws_url is not None:
            return _shared_ws_url
        proc = _serve_process(DEFAULT_PORT, cdp_max_connections=_CDP_MAX_CONNECTIONS)
        atexit.register(_kill_shared_proc)
        try:
            url = f"http://127.0.0.1:{DEFAULT_PORT}/json/version"
            ws_url: str | None = None
            async with httpx.AsyncClient() as c:
                for _ in range(120):
                    try:
                        r = await c.get(url, timeout=1)
                        ws_url = r.json()["webSocketDebuggerUrl"]
                        break
                    except Exception:
                        await asyncio.sleep(0.15)
            if ws_url is None:
                raise LightpandaError("Lightpanda CDP 服务未能及时就绪")
        except BaseException:
            _kill_shared_proc()
            raise
        _shared_proc = proc
        _shared_ws_url = ws_url
        _sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        return ws_url


async def close_shared() -> None:
    """Tear down the single shared serve process."""
    _kill_shared_proc()


async def _run_stateless(action, nav_timeout: int = DEFAULT_NAV_TIMEOUT) -> str:
    """Run action(engine, sid) on a fresh session attached to the shared server.

    Each call: acquire the concurrency semaphore → open its own ws + target →
    run → always close the ws + target. One flaky page can crash its own session
    without wedging others; on such a crash we retry once on a brand-new session.
    """
    ws_url = await _ensure_server()
    assert _sem is not None
    last: Exception | None = None
    async with _sem:
        for _ in (1, 2):
            # 会话**开始前**也看一眼：一轮里并发抓多个页面时（代理可以一次发十几个
            # web_fetch），子进程会在一次批量内迅速变大——只在收尾时检查就太晚了。
            _check_child_memory()
            eng = LightpandaEngine(nav_timeout=nav_timeout)
            sid: str | None = None
            try:
                await eng.attach(ws_url)
                sid = await eng.open_page()
                return await action(eng, sid)
            except (LightpandaError, websockets.ConnectionClosed, OSError) as e:
                last = e
            finally:
                if sid is not None:
                    try:
                        await eng.close_page(sid)
                    except Exception:
                        pass
                try:
                    await eng.close()  # not our process — just drops the ws
                except Exception:
                    pass
                _check_child_memory()
    # fail open：**原样**抛出真实错误（类型名 + 报文），不做原因揣测、也不给
    # "要不要走代理"之类的建议——那是 agent 的记忆与判断该做的事。
    # 以前这里写死"Lightpanda 仍为 Beta；繁重或 JS 框架页面可能使其崩溃"，
    # 把所有失败都归因成引擎脆弱，于是"外网不可达"被误读成"引擎不行"。
    raise LightpandaError(f"浏览器会话失败：{_err_text(last)}")


# ── plain-HTTP fast path (web_fetch) ─────────────────────────
#
# Many pages are static enough that opening a browser session is wasted work:
# GET the page, strip HTML → visible text. Only fall back to the browser engine
# when the fast path yields nothing usable (JS-rendered shells, non-HTML,
# network errors). Keeps web_fetch fast on the common path and usable even if
# Lightpanda isn't installed.

_FASTPATH_MIN_CHARS = 200
_FASTPATH_TIMEOUT = 12.0
_HTTP_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
_HTTP_ACCEPT_LANGUAGE = "zh-CN,zh;q=0.9,en;q=0.8"


class _HTMLToMarkdown(html.parser.HTMLParser):
    """Dependency-free HTML → Markdown.

    Emits headings (#), lists (-), links ([text](href) — the model can
    web_fetch them onward), simple tables (| cells) and fenced code blocks;
    like a browser, text data outside <pre> is whitespace-insensitive and
    collapsed. Page chrome (nav/header/footer/aside/forms/svg/…) and
    script/style are skipped; images are dropped entirely (token bloat +
    poison Feishu cards, see _strip_images).
    """

    _SKIP_TAGS = {
        "script", "style", "noscript", "template", "svg", "textarea",
        "nav", "header", "footer", "aside", "form", "select", "button",
        "input", "iframe", "object", "video", "audio", "meta", "link", "base",
    }
    # Void elements never emit an end tag — counting them in _skip would leak
    # and silently swallow the rest of the page.
    _VOID_TAGS = {
        "meta", "link", "base", "br", "img", "hr", "input", "wbr",
        "source", "area", "col", "embed", "track", "param",
    }
    _NEWLINE_TAGS = {
        "p", "div", "section", "article", "blockquote", "table", "ul",
        "ol", "dl", "tr", "figcaption", "dt", "dd",
    }
    _WS_RE = re.compile(r"[ \t\r\n]+")

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._skip = 0
        self._pre = 0
        self._in_title = False
        self._href: str | None = None
        self._link_text: list[str] = []
        self._chunks: list[str] = []
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            if tag not in self._VOID_TAGS:
                self._skip += 1
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            href = (dict(attrs).get("href") or "").strip()
            if self._href is None and href and not href.startswith(("javascript:", "#")):
                self._href = href
                self._link_text = []
        elif tag == "pre":
            self._pre += 1
            self._chunks.append("\n```\n")
        elif tag == "code" and not self._pre:
            self._chunks.append("`")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._chunks.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self._chunks.append("\n- ")
        elif tag in ("td", "th"):
            self._chunks.append("| ")
        elif tag == "br":
            self._chunks.append("\n")
        elif tag in self._NEWLINE_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            if tag not in self._VOID_TAGS and self._skip:
                self._skip -= 1
            return
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._href is not None:
            text = self._WS_RE.sub(" ", "".join(self._link_text)).strip()
            if text:
                href = self._href
                if self.base_url:
                    href = urllib.parse.urljoin(self.base_url, href)
                if href.startswith(("http://", "https://")):
                    self._chunks.append(f"[{text}]({href})")
                else:  # mailto:/ftp:/unresolvable — keep the text, drop the link
                    self._chunks.append(text)
            self._href = None
            self._link_text = []
        elif tag == "pre" and self._pre:
            self._pre -= 1
            self._chunks.append("\n```\n")
        elif tag == "code" and not self._pre:
            self._chunks.append("`")
        elif tag in ("td", "th"):
            self._chunks.append(" ")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data
        elif self._href is not None:
            self._link_text.append(data if self._pre else self._WS_RE.sub(" ", data))
        elif self._pre:
            self._chunks.append(data)
        else:
            self._chunks.append(self._WS_RE.sub(" ", data))

    def close(self) -> None:
        super().close()
        # Unclosed <a> — flush its accumulated text unlinked so it isn't lost.
        if self._href is not None:
            text = self._WS_RE.sub(" ", "".join(self._link_text)).strip()
            if text:
                self._chunks.append(text)
            self._href = None

    def markdown(self) -> str:
        return "".join(self._chunks)


def _html_to_markdown(html_text: str, base_url: str = "") -> str:
    parser = _HTMLToMarkdown(base_url=base_url)
    parser.feed(html_text)
    parser.close()
    # Empty list items ("-" alone) are icon/image bullets whose content was
    # dropped — pure noise.
    lines = [
        ln
        for ln in _squeeze_blanklines(
            _strip_images(parser.markdown(), cleanup_parens=False)
        ).split("\n")
        if ln.strip() != "-"
    ]
    body = "\n".join(lines).strip()
    title = parser.title.strip()
    if title and body:
        return f"# {title}\n\n{body}"
    return body or title


# ── page cache (web_fetch) ───────────────────────────────────
# 15-min TTL per URL. Stores the FULL uncapped markdown so different
# max_chars / prompt calls share one entry — the cap is applied per call after
# retrieval. The insertion-ordered dict doubles as the LRU list.

_FETCH_CACHE_TTL = 900.0
_FETCH_CACHE_MAX = 32
_fetch_cache: dict[str, tuple[float, str]] = {}


def _cache_get(url: str) -> str | None:
    hit = _fetch_cache.get(url)
    if hit is None:
        return None
    ts, text = hit
    if time.monotonic() - ts > _FETCH_CACHE_TTL:
        _fetch_cache.pop(url, None)
        return None
    return text


def _cache_put(url: str, text: str) -> None:
    _fetch_cache.pop(url, None)  # re-insert → freshest at the tail
    _fetch_cache[url] = (time.monotonic(), text)
    # 条数与**总字符数**双重封顶：只按条数的话，32 个繁重页面能占掉几个 G。
    while len(_fetch_cache) > _FETCH_CACHE_MAX or _cache_chars() > _FETCH_CACHE_MAX_CHARS:
        if len(_fetch_cache) <= 1:
            break
        _fetch_cache.pop(next(iter(_fetch_cache)))


def _cache_chars() -> int:
    return sum(len(t) for _, t in _fetch_cache.values())


def _err_text(e: BaseException) -> str:
    """异常的**真实**报文：str 为空时退到 __cause__/__context__（httpx 常把真因
    放在那里，比如 ConnectError 自身没报文、但 cause 是 DNS/覆盖层的具体错误）。

    只做转述，不做归因——错误是要暴露给 agent 的，由它凭记忆判断要不要换路。
    """
    parts: list[str] = []
    msg = str(e).strip()
    if msg:
        parts.append(msg)
    cause = e.__cause__ or e.__context__
    if cause is not None:
        cmsg = str(cause).strip() or repr(cause)
        if cmsg and cmsg not in parts:
            parts.append(cmsg)
    detail = " | ".join(parts) or "(无报文)"
    return f"{type(e).__name__}: {detail}"


async def http_fetch_once(url: str, timeout: float = _FASTPATH_TIMEOUT) -> tuple[str, str]:
    """Plain-HTTP page fetch → Markdown (no browser).

    返回 ``(text, reason)``：拿不到可用正文时 ``text`` 为空，``reason`` 说明**真实
    原因**（连接类异常 / HTTP 状态码 / 内容类型不对 / JS 壳页面）。过去这些都静默
    返回空串，结果是**最有诊断价值的连接层报错被吞掉**，只剩引擎那句含糊的
    websocket 报错——"外网不可达"于是被误读成"引擎太脆"。错误是要暴露给 agent 的，
    由它自己（凭记忆里的代理/镜像知识）决定怎么办。

    正文不在这一层封顶：调用方负责缓存整页并按次截断。
    """
    headers = {"User-Agent": _HTTP_USER_AGENT, "Accept-Language": _HTTP_ACCEPT_LANGUAGE}
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=timeout, headers=headers
        ) as client:
            resp = await client.get(url)
    except Exception as e:  # noqa: BLE001
        return "", _err_text(e)
    if resp.status_code >= 400:
        return "", f"HTTP {resp.status_code}"
    ctype = (resp.headers.get("content-type") or "").lower()
    if ctype and "html" not in ctype and "text/plain" not in ctype:
        return "", f"内容类型不是 HTML/文本（{ctype}）"
    try:
        text = _html_to_markdown(resp.text, base_url=str(resp.url))
    except Exception as e:  # noqa: BLE001
        return "", f"HTML 转 Markdown 失败：{type(e).__name__}: {str(e)[:120]}"
    # Tiny content WITH scripts is the classic JS-app shell signature — the
    # server sent a loader, the real text renders client-side. Hand those (and
    # empty results) to the browser engine; genuinely small static pages come
    # back as-is instead of paying for a browser session.
    if len(text.strip()) < _FASTPATH_MIN_CHARS and "<script" in resp.text.lower():
        return "", f"JS 壳页面（正文仅 {len(text.strip())} 字且含 script）"
    return _cap_page(text), ""


async def fetch_once(
    url: str,
    max_chars: int = 8000,
    http_timeout: float = _FASTPATH_TIMEOUT,
    no_cache: bool = False,
) -> str:
    """Stateless page fetch → Markdown.

    Plain-HTTP fast path first (most static pages need no browser), then a real
    browser session for JS-rendered pages. The full-page markdown is cached per
    URL (15-min TTL); `max_chars` caps only the returned text, so a cached page
    serves different caps / prompts without refetching.
    """
    if not no_cache:
        cached = _cache_get(url)
        if cached is not None:
            return _cap(cached, max_chars)

    text, fast_reason = await http_fetch_once(url, timeout=http_timeout)
    if not text.strip():

        async def _do(eng: LightpandaEngine, sid: str) -> str:
            await eng.navigate(sid, url)
            return await eng.read_markdown(sid)

        try:
            text = await _run_stateless(_do, nav_timeout=int(http_timeout))
        except LightpandaError as engine_error:
            # fail open：把两条**真实错误**一起交出去（直连报了什么、浏览器报了什么），
            # 不揣测原因、不建议走代理。agent 凭自己的记忆判断要不要换路。
            raise LightpandaError(f"直连：{fast_reason or '无可用正文'}；{engine_error}") from engine_error

    if text.strip() and not no_cache:
        _cache_put(url, text)
    return _cap(text, max_chars)


# Bing renders enough server-side for Lightpanda's DOM engine (verified);
# DuckDuckGo's endpoints crash it. The PRIMARY search path is the RSS endpoint
# below — plain HTTP, structured items, immune to Bing page redesigns. The
# real browser session over the HTML results page is only the fallback.
_SEARCH_URL = "https://www.bing.com/search?q={q}"
_SEARCH_RSS_URL = "https://www.bing.com/search?q={q}&format=rss&count={n}"


def _parse_bing_rss(xml_text: str) -> list[dict[str, str]]:
    """Bing RSS 2.0 XML → [{'title','url','snippet'}].

    Tolerant by design: returns [] on anything unparseable (error pages,
    endpoint changes) so the caller falls back to the browser session.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    results: list[dict[str, str]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue
        # Descriptions occasionally carry markup / nested entities — strip
        # tags first, then unescape what's left, then squeeze whitespace.
        desc = re.sub(r"<[^>]+>", " ", html.unescape(item.findtext("description") or ""))
        results.append({
            "title": title,
            "url": link,
            "snippet": re.sub(r"\s+", " ", desc).strip(),
        })
    return results


async def _search_rss(query: str, count: int, timeout: float) -> list[dict[str, str]]:
    """Fetch Bing's RSS search endpoint over plain HTTP (no browser).

    www.bing.com 302s to the regional host (e.g. cn.bing.com) — follow_redirects
    handles it. 返回 ``(results, reason)``：失败时 reason 是真实报错（不静默吞），
    由调用方决定兜底并**把真因暴露给 agent**。
    """
    url = _SEARCH_RSS_URL.format(q=urllib.parse.quote(query), n=max(count, 10))
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": _HTTP_USER_AGENT, "Accept-Language": _HTTP_ACCEPT_LANGUAGE},
        ) as client:
            resp = await client.get(url)
    except Exception as e:  # noqa: BLE001
        return [], _err_text(e)
    if resp.status_code >= 400:
        return [], f"HTTP {resp.status_code}"
    items = _parse_bing_rss(resp.text)
    return items, ("" if items else "RSS 解析无结果（端点变更或返回错误页）")


def _domain_allowed(url: str, allowed: list[str] | None, blocked: list[str] | None) -> bool:
    """Domain filter for structured results (exact-host or dot-suffix match)."""
    if not allowed and not blocked:
        return True
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if not host:
        return not allowed  # opaque URL — only let it through when unfiltered

    def _match(domains: list[str]) -> bool:
        return any(host == d.lower() or host.endswith("." + d.lower()) for d in domains)

    if blocked and _match(blocked):
        return False
    if allowed and not _match(allowed):
        return False
    return True


def _render_results(results: list[dict[str, str]]) -> str:
    """Numbered markdown-link list: `N. [title](url)` + indented snippet."""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['title']}]({r['url']})")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
    return "\n".join(lines)


async def search_once(
    query: str,
    max_chars: int = 8000,
    timeout: float = _FASTPATH_TIMEOUT,
    max_results: int = 8,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> str:
    """Stateless web search.

    Primary: Bing's RSS endpoint over plain HTTP — structured results, no
    browser. Fallback: a real browser session over the HTML results page
    (crash-recovering). Domain filters apply to the structured RSS results.
    """
    max_results = max(1, min(int(max_results), 30))
    rss, rss_reason = await _search_rss(query, count=max_results * 3, timeout=timeout)
    if rss:
        hits = [r for r in rss if _domain_allowed(r["url"], allowed_domains, blocked_domains)]
        if hits:
            return _cap(_render_results(hits[:max_results]), max_chars)
        # RSS worked but the filters eliminated everything — a browser session
        # would return the same unfiltered list, so don't pay for one.
        return "没有符合域名过滤条件的搜索结果。"

    url = _SEARCH_URL.format(q=urllib.parse.quote(query))

    async def _do(eng: LightpandaEngine, sid: str) -> str:
        await eng.navigate(sid, url)
        # Drop Bing's header nav and footer; keep just the result list.
        return _trim_bing_chrome(await eng.read_markdown(sid))

    try:
        return _cap(await _run_stateless(_do, nav_timeout=int(timeout)), max_chars)
    except LightpandaError as engine_error:
        raise LightpandaError(f"Bing RSS：{rss_reason or '无结果'}；{engine_error}") from engine_error
