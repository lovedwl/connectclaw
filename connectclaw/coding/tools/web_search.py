"""
Web search & fetch — backed by the Lightpanda browser engine (see lightpanda.py).

Replaces the old glyph TUI scrape: a real (headless) browser session, Bing for
search, DOM→text extraction, crash-recovering. No HTML/JS/ads → token-efficient.

For login / multi-step interaction, use the stateful `browser` tool instead;
these two are the stateless, main-agent-facing shortcuts. Both share the same
Lightpanda engine under the hood.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.coding.tools import lightpanda
from connectclaw.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CHARS = 8000
DEFAULT_TIMEOUT = 30


@dataclass
class WebSearchConfig:
    max_chars: int = DEFAULT_MAX_CHARS
    timeout: int = DEFAULT_TIMEOUT


# ── Web Search Tool ──────────────────────────────────────────────


class WebSearchTool(AgentTool):
    name = "web_search"
    label = "web_search"
    description = (
        "Search the web and get PLAIN TEXT results via a real headless browser "
        "session (Bing). Token-efficient: clean text, no HTML/ads. "
        "For fetching a specific URL use web_fetch; for login or multi-step "
        "interaction use the `browser` tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
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
        max_chars = int(params.get("max_chars") or self._config.max_chars)
        try:
            text = await lightpanda.search_once(query, max_chars)
            return AgentToolResult(
                content=[{"type": "text", "text": text}],
                details={"query": query, "chars": len(text)},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("web_search failed: %s", e)
            return AgentToolResult(
                content=[{"type": "text", "text": f"Web search failed: {e}"}],
            )


# ── Web Fetch Tool ───────────────────────────────────────────────


class WebFetchTool(AgentTool):
    name = "web_fetch"
    label = "web_fetch"
    description = (
        "Fetch a URL as PLAIN TEXT via a real headless browser session. "
        "Strips HTML/JS/ads — token-efficient. Best for articles, docs, blogs. "
        "Very heavy SPAs may fail (engine is lightweight); use the `browser` "
        "tool for interactive/JS-heavy pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL to fetch"},
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
        max_chars = int(params.get("max_chars") or self._config.max_chars)
        try:
            text = await lightpanda.fetch_once(url, max_chars)
            return AgentToolResult(
                content=[{"type": "text", "text": text}],
                details={"url": url, "chars": len(text)},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("web_fetch failed: %s", e)
            return AgentToolResult(
                content=[{"type": "text", "text": f"Web fetch failed: {e}"}],
            )


# ── Factory Functions ────────────────────────────────────────────


def create_web_search_tool(
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = DEFAULT_TIMEOUT,
) -> WebSearchTool:
    return WebSearchTool(config=WebSearchConfig(max_chars=max_chars, timeout=timeout))


def create_web_fetch_tool(
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = DEFAULT_TIMEOUT,
) -> WebFetchTool:
    return WebFetchTool(config=WebSearchConfig(max_chars=max_chars, timeout=timeout))
