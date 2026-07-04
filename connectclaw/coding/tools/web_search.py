"""
Web search & fetch tools powered by the `glyph` TUI browser.

Uses `glyph --dump` for both search and page fetch.
Search is backed by Bing (via glyph's Template engine).
Output is clean plain text — no HTML, no JS, no ads.
This saves ~8x tokens compared to raw HTML scraping.

Reference: https://github.com/k1y0miiii/glyph
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CHARS = 8000
DEFAULT_TIMEOUT = 30

# ConnectClaw's own glyph profile — uses Bing as search engine.
_GLYPH_PROFILE_DIR = os.path.expanduser("~/.connectclaw")
_GLYPH_PROFILE_PATH = os.path.join(_GLYPH_PROFILE_DIR, "glyph_profile.json")

_BING_TEMPLATE = "https://www.bing.com/search?q={q}"


@dataclass
class WebSearchConfig:
    glyph_bin: str = "glyph"
    max_chars: int = DEFAULT_MAX_CHARS
    timeout: int = DEFAULT_TIMEOUT


# ── Glyph Profile ────────────────────────────────────────────────


def _ensure_glyph_profile() -> str:
    """Create (if missing) a ConnectClaw-specific glyph profile with Bing search.

    Returns the path to the profile file.
    """
    os.makedirs(_GLYPH_PROFILE_DIR, exist_ok=True)
    if not os.path.isfile(_GLYPH_PROFILE_PATH):
        profile = {
            "settings": {
                "search_engine": {"Template": _BING_TEMPLATE},
                "gemini_model": "gemini-2.5-flash",
                "gemini_api_key": "",
            },
            "history": [],
            "bookmarks": [],
            "cookies": [],
            "chats": [],
            "session": {
                "tabs": [{"url": "glyph:home", "chat_id": None}],
                "active": 0,
            },
        }
        with open(_GLYPH_PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=2)
        logger.info("Created glyph profile at %s (Bing search)", _GLYPH_PROFILE_PATH)
    return _GLYPH_PROFILE_PATH


# ── Glyph Subprocess ────────────────────────────────────────────


def _find_glyph(glyph_bin: str) -> str | None:
    """Resolve the glyph binary path.

    Tries in order:
      1. The configured path (absolute or bare name on PATH).
      2. ~/.cargo/bin/glyph (default cargo install location).
    Returns the resolved path, or None if not found.
    """
    if os.path.isabs(glyph_bin) and os.path.isfile(glyph_bin):
        return glyph_bin
    if shutil.which(glyph_bin):
        return glyph_bin
    cargo_glyph = os.path.expanduser("~/.cargo/bin/glyph")
    if os.path.isfile(cargo_glyph):
        return cargo_glyph
    return None


async def _run_glyph(
    args: list[str],
    glyph_bin: str = "glyph",
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Run glyph with arguments, return stdout as string.

    Sets GLYPH_PROFILE_PATH to ConnectClaw's Bing profile so search
    always uses Bing regardless of the user's interactive glyph settings.
    """
    bin_path = _find_glyph(glyph_bin)
    if bin_path is None:
        raise RuntimeError(
            "glyph not found. Install it with:\n"
            "  git clone https://github.com/k1y0miiii/glyph.git\n"
            "  cd glyph && cargo install --locked --path crates/app\n"
            "Or set GLYPH_BIN env var / glyph_bin in config.toml."
        )

    env = os.environ.copy()
    env["GLYPH_PROFILE_PATH"] = _ensure_glyph_profile()

    try:
        proc = await asyncio.create_subprocess_exec(
            bin_path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"glyph timed out after {timeout}s — try a narrower query.")
    except FileNotFoundError:
        raise RuntimeError(
            f"glyph binary not found at '{bin_path}'. "
            "Install: git clone https://github.com/k1y0miiii/glyph.git && "
            "cd glyph && cargo install --locked --path crates/app"
        )
    except Exception as e:
        raise RuntimeError(f"glyph execution failed: {e}")

    if proc.returncode != 0:
        err_text = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"glyph exited with code {proc.returncode}: {err_text}")

    return stdout.decode("utf-8", errors="replace")


# ── Output Helpers ───────────────────────────────────────────────


def _cap(text: str, max_chars: int) -> str:
    """Truncate text to max_chars with a note."""
    if len(text) <= max_chars:
        return text
    cutoff = f"\n…[truncated to {max_chars} chars — narrow your query or raise max_chars]"
    return text[:max_chars - len(cutoff)] + cutoff


def _clean_search(dump: str) -> str:
    """Strip Bing search chrome — header nav, footer pagination/legal.

    Bing pages rendered by glyph have a predictable structure:
      - Header chrome: nav links, search bar, tabs, filter bar
      - Results: numbered items (\"  N.\") with domain, URL, title, separator, snippet
      - Footer chrome: legal notice, pagination, license text

    We slice from the first numbered result to the last real result,
    dropping the header and footer.
    """
    import re

    lines = dump.split("\n")

    # Find where results start: first line matching "  N." (indented number + dot)
    first_result = None
    for i, line in enumerate(lines):
        if re.match(r"^\s+\d+\.\s*$", line):
            first_result = i
            break

    if first_result is None:
        return _collapse(dump)

    # Find the range of each numbered result block (from "  N." to before next "  N.")
    result_starts: list[int] = []
    for i, line in enumerate(lines):
        if re.match(r"^\s+\d+\.\s*$", line):
            result_starts.append(i)

    # Determine which result blocks are real (contain a URL) vs footer (no URL)
    last_real_idx = None
    for idx, start in enumerate(result_starts):
        end = result_starts[idx + 1] if idx + 1 < len(result_starts) else len(lines)
        block = "\n".join(lines[start:end])
        if re.search(r"https?://", block):
            last_real_idx = idx

    if last_real_idx is None:
        return _collapse(dump)

    # Slice from first real result to end of last real result block
    begin = result_starts[0]
    end = result_starts[last_real_idx + 1] if last_real_idx + 1 < len(result_starts) else len(lines)

    # Additionally trim known footer keywords after the last real URL
    last_url_line = None
    for i in range(begin, end):
        if re.search(r"https?://", lines[i]):
            last_url_line = i

    footer_kw = (
        "分页", "增值电信", "京ICP", "京公网", "隐私", "条款",
        "全部", "下一页", "为回应", "法律要求", "此处", "未予显示",
    )
    for i in range((last_url_line or begin) + 1, end):
        if any(kw in lines[i] for kw in footer_kw):
            end = i
            break

    body = lines[begin:end]
    return _collapse("\n".join(body))


def _clean_fetch(dump: str) -> str:
    """Collapse excessive blank lines for fetched pages."""
    return _collapse(dump)


def _collapse(text: str) -> str:
    """Normalize line endings: replace 3+ consecutive newlines with 2."""
    import re
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ── Web Search Tool ──────────────────────────────────────────────


class WebSearchTool(AgentTool):
    name = "web_search"
    label = "web_search"
    description = (
        "Search the web and get PLAIN TEXT results via the glyph browser "
        "(Bing engine). Token-efficient: returns clean text without HTML. "
        "Best for general, text, and documentation queries. "
        "For fetching a specific URL, use web_fetch instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_chars": {
                "type": "integer",
                "description": f"Max characters in the response (default: {DEFAULT_MAX_CHARS})",
            },
        },
        "required": ["query"],
    }

    def __init__(self, config: WebSearchConfig):
        self._config = config

    async def execute(
        self,
        tool_call_id: str,
        params: dict,
        signal: asyncio.Event | None = None,
        on_update: object = None,
    ) -> AgentToolResult:
        query = params["query"]
        max_chars = params.get("max_chars") or self._config.max_chars

        try:
            dump = await _run_glyph(
                ["--dump", query],
                glyph_bin=self._config.glyph_bin,
                timeout=self._config.timeout,
            )
            text = _cap(_clean_search(dump), int(max_chars))
            return AgentToolResult(
                content=[{"type": "text", "text": text}],
                details={"query": query, "chars": len(text)},
            )
        except RuntimeError as e:
            logger.warning("web_search failed: %s", e)
            return AgentToolResult(
                content=[{"type": "text", "text": f"Web search failed: {e}"}],
            )


# ── Web Fetch Tool ───────────────────────────────────────────────


class WebFetchTool(AgentTool):
    name = "web_fetch"
    label = "web_fetch"
    description = (
        "Fetch a URL as PLAIN TEXT via the glyph browser. "
        "Strips HTML, JS, navigation, and ads — token-efficient. "
        "Best for articles, documentation, blogs, and static/text pages. "
        "NOT suitable for JS-heavy SPAs or dynamic dashboards (returns little content) — "
        "use an alternative fetch method for those."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL to fetch",
            },
            "max_chars": {
                "type": "integer",
                "description": f"Max characters in the response (default: {DEFAULT_MAX_CHARS})",
            },
        },
        "required": ["url"],
    }

    def __init__(self, config: WebSearchConfig):
        self._config = config

    async def execute(
        self,
        tool_call_id: str,
        params: dict,
        signal: asyncio.Event | None = None,
        on_update: object = None,
    ) -> AgentToolResult:
        url = params["url"]
        max_chars = params.get("max_chars") or self._config.max_chars

        try:
            dump = await _run_glyph(
                ["--dump", url],
                glyph_bin=self._config.glyph_bin,
                timeout=self._config.timeout,
            )
            text = _cap(_clean_fetch(dump), int(max_chars))
            return AgentToolResult(
                content=[{"type": "text", "text": text}],
                details={"url": url, "chars": len(text)},
            )
        except RuntimeError as e:
            logger.warning("web_fetch failed: %s", e)
            return AgentToolResult(
                content=[{"type": "text", "text": f"Web fetch failed: {e}"}],
            )


# ── Factory Functions ────────────────────────────────────────────


def create_web_search_tool(
    glyph_bin: str = "glyph",
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = DEFAULT_TIMEOUT,
) -> WebSearchTool:
    return WebSearchTool(
        config=WebSearchConfig(
            glyph_bin=glyph_bin,
            max_chars=max_chars,
            timeout=timeout,
        )
    )


def create_web_fetch_tool(
    glyph_bin: str = "glyph",
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = DEFAULT_TIMEOUT,
) -> WebFetchTool:
    return WebFetchTool(
        config=WebSearchConfig(
            glyph_bin=glyph_bin,
            max_chars=max_chars,
            timeout=timeout,
        )
    )
