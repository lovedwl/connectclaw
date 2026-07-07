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
import json
import os
import re
import urllib.parse
from typing import Any

import httpx
import websockets

DEFAULT_PORT = 9222
DEFAULT_NAV_TIMEOUT = 20          # s, page navigation budget
_SERVE_INACTIVITY = 3600          # s, keep CDP alive for long-lived sessions
_RECV_SLACK = 15                  # s added to nav timeout for ws recv


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
            self._ws = await websockets.connect(ws_url, max_size=None, open_timeout=15)
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
        self._ws = await websockets.connect(ws_url, max_size=None, open_timeout=15)

    async def _await_ready(self) -> str:
        url = f"http://127.0.0.1:{self._port}/json/version"
        async with httpx.AsyncClient() as c:
            for _ in range(120):
                try:
                    r = await c.get(url, timeout=1)
                    return r.json()["webSocketDebuggerUrl"]
                except Exception:
                    await asyncio.sleep(0.15)
        raise LightpandaError("Lightpanda CDP server did not become ready in time")

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
            raise LightpandaError("engine not started")
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
                return _squeeze_blanklines(_strip_images(md).strip())
        except LightpandaError:
            pass  # command missing / page too heavy — fall back below

        raw = await self._eval(sid, _EXTRACT_JS)
        try:
            data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except (json.JSONDecodeError, TypeError):
            data = {"title": "", "text": str(raw or "")}
        title = (data.get("title") or "").strip()
        text = _collapse(data.get("text") or "")
        return f"# {title}\n\n{text}" if title else text

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


def _strip_images(text: str) -> str:
    text = _IMG_MD_RE.sub("", text)
    text = _EMPTY_LINK_RE.sub("", text)
    # Tidy up the "()" / " ." leftovers a removed inline element can leave behind.
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
    note = f"\n…[truncated to {max_chars} chars]"
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
                raise LightpandaError("Lightpanda CDP server did not become ready in time")
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


async def _run_stateless(action) -> str:
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
            eng = LightpandaEngine()
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
    raise LightpandaError(
        "browser engine failed on this page (Lightpanda is Beta; heavy or "
        f"JS-framework pages can crash it): {last}"
    )


async def fetch_once(url: str, max_chars: int = 8000) -> str:
    """Stateless: open a fresh page, navigate, read markdown. Crash-recovering."""
    async def _do(eng: LightpandaEngine, sid: str) -> str:
        await eng.navigate(sid, url)
        return _cap(await eng.read_markdown(sid), max_chars)

    return await _run_stateless(_do)


# Bing renders enough server-side for Lightpanda's DOM engine (verified);
# DuckDuckGo's endpoints crash it. A real browser session beats glyph's scrape.
_SEARCH_URL = "https://www.bing.com/search?q={q}"


async def search_once(query: str, max_chars: int = 8000) -> str:
    """Stateless web search via a real browser session (Bing). Crash-recovering."""
    url = _SEARCH_URL.format(q=urllib.parse.quote(query))

    async def _do(eng: LightpandaEngine, sid: str) -> str:
        await eng.navigate(sid, url)
        md = await eng.read_markdown(sid)
        # Drop Bing's header nav and footer; keep just the result list.
        return _cap(_trim_bing_chrome(md), max_chars)

    return await _run_stateless(_do)
