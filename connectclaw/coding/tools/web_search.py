"""
Web search tool — sub-agent that searches the web and summarizes results.

Uses Bing Web Search API as backend. The sub-agent pattern means:
1. Main agent calls web_search tool with a query
2. Tool spawns a sub-agent to search + summarize
3. Result is returned to main agent as tool result
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.provider.types import Context, Model, UserMessage


@dataclass
class WebSearchConfig:
    api_key: str = ""
    endpoint: str = "https://api.bing.microsoft.com/v7.0/search"
    max_results: int = 5
    timeout: int = 30


class WebSearchTool(AgentTool):
    name = "web_search"
    label = "web_search"
    description = (
        "Search the web for information. "
        "Use this to find current information, documentation, or answers "
        "that require looking up external sources. "
        "Returns summarized results from web pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 10)",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        config: WebSearchConfig,
        model: Model | None = None,
        api_key: str | None = None,
    ):
        self._config = config
        self._model = model
        self._api_key = api_key

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        query = params["query"]
        num_results = min(params.get("num_results", 5) or 5, 10)

        # If no API key configured, return placeholder
        if not self._config.api_key:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"Web search is not configured (BING_API_KEY not set). "
                        f"Query would have been: '{query}'\n\n"
                        f"To enable web search, set BING_API_KEY in .env or config.toml."
                    ),
                }],
            )

        try:
            # Step 1: Perform Bing search
            import urllib.request
            import urllib.parse

            encoded_query = urllib.parse.quote(query)
            url = f"{self._config.endpoint}?q={encoded_query}&count={num_results}"

            req = urllib.request.Request(url)
            req.add_header("Ocp-Apim-Subscription-Key", self._config.api_key)

            response = await asyncio.to_thread(
                urllib.request.urlopen, req, timeout=self._config.timeout
            )
            data = json.loads(response.read().decode())

            # Step 2: Extract results
            results = []
            for item in data.get("webPages", {}).get("value", [])[:num_results]:
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                })

            if not results:
                return AgentToolResult(
                    content=[{"type": "text", "text": f"No results found for: {query}"}],
                )

            # Step 3: Format results
            output_lines = [f"Search results for: **{query}**\n"]
            for i, r in enumerate(results, 1):
                output_lines.append(f"### {i}. {r['title']}")
                output_lines.append(f"URL: {r['url']}")
                output_lines.append(f"{r['snippet']}\n")

            return AgentToolResult(
                content=[{"type": "text", "text": "\n".join(output_lines)}],
                details={"results": results, "query": query},
            )

        except Exception as e:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": f"Web search failed: {e}",
                }],
            )


def create_web_search_tool(
    api_key: str = "",
    model: Model | None = None,
) -> WebSearchTool:
    return WebSearchTool(
        config=WebSearchConfig(api_key=api_key),
        model=model,
        api_key=api_key,
    )
