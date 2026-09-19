"""
Web search & fetch tools.

Search: Bing's RSS endpoint over plain HTTP for structured, numbered results
(fast, redesign-proof), falling back to a real headless browser session.
No HTML/JS/ads → token-efficient.

Fetch: plain-HTTP fast path for static pages (no browser needed), falling back
to a browser session for JS-rendered pages. Optional `prompt`: a one-shot LLM
reads the page and answers directly, so the main model gets the answer instead
of the whole page. Pages are cached ~15 min per URL.

These two are the stateless, main-agent-facing shortcuts, sharing one Lightpanda
engine under the hood. (A stateful `browser` tool once existed for login /
multi-step interaction but has been retired.)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.coding.tools import lightpanda
from connectclaw.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CHARS = 8000
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RESULTS = 8

# ── prompt extraction (web_fetch) ────────────────────────────────
# Optional `prompt` mode: a one-shot LLM reads the fetched page and answers,
# so only the answer enters the main model's context, not the whole page.
_EXTRACT_SYSTEM = (
    "You are a web-page reading assistant. Answer the user's question using "
    "ONLY the provided page content; if the page doesn't contain the answer, "
    "say so plainly instead of guessing. Be concise and concrete (facts, "
    "numbers, quotes help). Reply in the language of the question."
)
_EXTRACT_MAX_PAGE_CHARS = 30000
_EXTRACT_MAX_TOKENS = 2000


@dataclass
class WebSearchConfig:
    max_chars: int = DEFAULT_MAX_CHARS
    timeout: int = DEFAULT_TIMEOUT
    # models.toml profile name for web_fetch `prompt` extraction (a cheap small
    # model). Empty = follow the active main model (hot-switches with /model).
    extract_model: str = ""


# ── Web Search Tool ──────────────────────────────────────────────


class WebSearchTool(AgentTool):
    name = "web_search"
    label = "web_search"
    description = (
        "Search the web (Bing). Returns numbered results as markdown links: "
        "`1. [title](url)` followed by a snippet, so you can web_fetch any of "
        "them. Supports max_results and allowed_domains/blocked_domains "
        "filtering. Token-efficient: clean text, no HTML/ads."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": f"Max number of results (default: {DEFAULT_MAX_RESULTS}, max 30)",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'Only return results from these domains, e.g. ["python.org"]',
            },
            "blocked_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exclude results from these domains",
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
        max_chars = int(params.get("max_chars") or self._config.max_chars)
        try:
            text = await lightpanda.search_once(
                query,
                max_chars,
                timeout=float(self._config.timeout),
                max_results=int(params.get("max_results") or DEFAULT_MAX_RESULTS),
                allowed_domains=params.get("allowed_domains") or None,
                blocked_domains=params.get("blocked_domains") or None,
            )
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
        "Fetch a URL as PLAIN MARKDOWN. Static pages use a fast plain-HTTP "
        "path; JS-rendered pages fall back to a real headless browser session. "
        "Pass `prompt` to have a fast LLM read the page and answer your "
        "question directly — only the answer enters context, not the whole "
        "page (use for 'what does this page say about X' lookups). Results "
        "are cached ~15 min per URL (no_cache bypasses)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL to fetch"},
            "prompt": {
                "type": "string",
                "description": (
                    "Optional question to answer from the page via a one-shot "
                    "LLM call. Omit to get the raw page markdown instead."
                ),
            },
            "max_chars": {
                "type": "integer",
                "description": f"Max characters in the response (default: {DEFAULT_MAX_CHARS})",
            },
            "no_cache": {
                "type": "boolean",
                "description": "Skip the ~15-min page cache and refetch",
            },
        },
        "required": ["url"],
    }

    def __init__(
        self,
        config: WebSearchConfig,
        *,
        model_provider: Callable[[], Any] | None = None,
        api_key_provider: Callable[[], str] | None = None,
        proxy: str = "",
    ):
        self._config = config
        # Callables, not objects: resolved at execute() time so /model
        # hot-switches are picked up (coding_agent swaps its live Model).
        self._model_provider = model_provider
        self._api_key_provider = api_key_provider
        self._proxy = proxy

    async def execute(
        self,
        tool_call_id: str,
        params: dict,
        signal: asyncio.Event | None = None,
        on_update: object = None,
    ) -> AgentToolResult:
        url = params["url"]
        prompt = str(params.get("prompt") or "").strip()
        max_chars = int(params.get("max_chars") or self._config.max_chars)
        no_cache = bool(params.get("no_cache"))
        try:
            text = await lightpanda.fetch_once(
                url,
                max_chars,
                http_timeout=float(self._config.timeout),
                no_cache=no_cache,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("web_fetch failed: %s", e)
            return AgentToolResult(
                content=[{"type": "text", "text": f"Web fetch failed: {e}"}],
            )

        if not prompt:
            return AgentToolResult(
                content=[{"type": "text", "text": text}],
                details={"url": url, "chars": len(text)},
            )

        answer = await self._extract_answer(prompt, text)
        if answer is not None:
            return AgentToolResult(
                content=[{"type": "text", "text": answer}],
                details={"url": url, "prompt": prompt, "extracted": True},
            )
        # Extraction unavailable or failed — degrade to the raw page, marked.
        note = "(prompt extraction unavailable — raw page content returned)"
        return AgentToolResult(
            content=[{"type": "text", "text": f"{note}\n\n{text}"}],
            details={"url": url, "chars": len(text), "extracted": False},
        )

    # ── prompt extraction ────────────────────────────────────────

    def _resolve_endpoint(self) -> tuple[str, str, str, str] | None:
        """(base_url, model_id, api_key, proxy) for the extraction call, or
        None when no usable model is available.

        Preferred: the configured models.toml profile (cheap small model).
        Otherwise: the live main model via the injected callables — resolved
        per call so /model hot-switches are honored.
        """
        if self._config.extract_model:
            try:
                from connectclaw.model_registry import ModelsStore

                profile = ModelsStore().get(self._config.extract_model)
            except Exception:  # noqa: BLE001
                profile = None
            if profile is not None and profile.base_url and profile.model_id:
                return (
                    profile.base_url,
                    profile.model_id,
                    profile.resolved_api_key(),
                    self._proxy,  # profiles don't carry a proxy; model APIs ride [proxy]
                )
            logger.warning(
                "web_fetch extract_model %r not found in models.toml — using the main model",
                self._config.extract_model,
            )
        model = self._model_provider() if self._model_provider else None
        if model is None or not getattr(model, "base_url", None):
            return None
        api_key = self._api_key_provider() if self._api_key_provider else ""
        return model.base_url, model.id, api_key, getattr(model, "proxy", "") or self._proxy

    async def _extract_answer(self, prompt: str, page_text: str) -> str | None:
        """One-shot LLM: answer `prompt` from the page text. None on any
        failure — the caller degrades to returning the raw page."""
        endpoint = self._resolve_endpoint()
        if endpoint is None:
            return None
        base_url, model_id, api_key, proxy = endpoint
        if not api_key:
            return None
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return None

        question = (
            f"{prompt}\n\n---\n\nBelow is the content of a web page. Answer the "
            f"question above using only this content.\n\n"
            f"{page_text[:_EXTRACT_MAX_PAGE_CHARS]}"
        )
        # Same pattern as image_analyze: non-streaming OpenAI-compatible call;
        # model APIs ride [proxy] with trust_env=False (ignore ambient env).
        http_client: httpx.AsyncClient | None = None
        try:
            if proxy:
                http_client = httpx.AsyncClient(proxy=proxy, trust_env=False)
                client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user", "content": question},
                ],
                max_tokens=_EXTRACT_MAX_TOKENS,
            )
            text = (response.choices[0].message.content or "").strip()
            return text or None
        except Exception as e:  # noqa: BLE001
            logger.warning("web_fetch prompt extraction failed: %s", e)
            return None
        finally:
            if http_client is not None:
                await http_client.aclose()


# ── Factory Functions ────────────────────────────────────────────


def create_web_search_tool(
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = DEFAULT_TIMEOUT,
) -> WebSearchTool:
    return WebSearchTool(config=WebSearchConfig(max_chars=max_chars, timeout=timeout))


def create_web_fetch_tool(
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = DEFAULT_TIMEOUT,
    extract_model: str = "",
    model_provider: Callable[[], Any] | None = None,
    api_key_provider: Callable[[], str] | None = None,
    proxy: str = "",
) -> WebFetchTool:
    return WebFetchTool(
        config=WebSearchConfig(
            max_chars=max_chars, timeout=timeout, extract_model=extract_model
        ),
        model_provider=model_provider,
        api_key_provider=api_key_provider,
        proxy=proxy,
    )
