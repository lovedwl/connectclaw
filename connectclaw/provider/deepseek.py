"""OpenAI-compatible LLM provider."""

import base64
import json
import os
from functools import lru_cache

from openai import AsyncOpenAI

from .types import (
    AssistantMessage,
    Model,
    ToolDef,
)


@lru_cache(maxsize=64)
def _read_base64(path: str, mtime_ns: int) -> str:
    """Cache base64 of an attachment file, keyed by (path, mtime)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


class DeepSeekProvider:
    """OpenAI-compatible provider with reasoning_content + multimodal support."""

    def __init__(self, base_url: str = "https://api.deepseek.com"):
        self.base_url = base_url

    def build_client(
        self,
        api_key: str,
        base_url: str | None = None,
        proxy: str | None = None,
    ) -> AsyncOpenAI:
        if proxy:
            import httpx

            http_client = httpx.AsyncClient(proxy=proxy, trust_env=False)
            return AsyncOpenAI(
                base_url=base_url or self.base_url,
                api_key=api_key,
                http_client=http_client,
            )
        return AsyncOpenAI(base_url=base_url or self.base_url, api_key=api_key)

    def convert_tools(self, tools: list[ToolDef] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def convert_messages(self, messages: list) -> list[dict]:
        """Convert ConnectClaw messages to OpenAI format.

        Image attach policy (prefix-cache discipline): image_ref blocks are
        resolved to inline ``image_url`` data URLs only for the current turn —
        the last user message and anything after it (attach_image tool
        results). Image blocks in older history degrade to a short text
        placeholder so base64 never sits in the long-lived text prefix that
        the provider's prefix cache is keyed on; images are billed per turn,
        so keeping them out of history keeps every turn cheap and the prefix
        byte-stable.
        """
        last_user_idx = -1
        for i, m in enumerate(messages):
            if m.role == "user":
                last_user_idx = i

        result: list[dict] = []
        for i, m in enumerate(messages):
            resolve = i >= last_user_idx
            if m.role == "user":
                result.append(self._convert_user_message(m, resolve))
            elif m.role == "assistant":
                result.append(self._convert_assistant_message(m))
            elif m.role == "toolResult":
                tool_msg, image_parts = self._convert_tool_result(m, resolve)
                result.append(tool_msg)
                # Attached images ride in a synthetic *user* message — tool
                # messages are text-only in the OpenAI-compatible wire format.
                if image_parts:
                    result.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "[已挂载图片]"}, *image_parts],
                    })
        return result

    # ── Per-role conversion ───────────────────────────────────

    def _convert_user_message(self, m, resolve: bool = True) -> dict:
        content = m.content
        if isinstance(content, str):
            return {"role": "user", "content": content}
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append({"type": "text", "text": str(block)})
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif btype in ("image", "image_ref"):
                converted = self._convert_image_block(block, resolve)
                if converted is not None:
                    parts.append(converted)
        return {"role": "user", "content": parts}

    def _convert_assistant_message(self, m) -> dict:
        msg: dict = {"role": "assistant", "content": ""}
        text_parts = []
        tool_calls = []
        thinking_parts = []

        for block in m.content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif btype == "toolCall":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("arguments", {})),
                    },
                })

        msg["content"] = "\n".join(text_parts) or None
        if thinking_parts:
            msg["reasoning_content"] = "\n".join(thinking_parts)
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def _convert_tool_result(self, m, resolve: bool = True) -> tuple[dict, list[dict]]:
        """Convert a toolResult to a text-only tool message plus resolved images."""
        text = ""
        image_parts: list[dict] = []
        for block in m.content:
            if not isinstance(block, dict):
                text += str(block)
                continue
            btype = block.get("type")
            if btype == "text":
                text += block.get("text", "")
            elif btype in ("image", "image_ref"):
                converted = self._convert_image_block(block, resolve)
                if converted is not None and converted.get("type") == "image_url":
                    image_parts.append(converted)
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": text,
        }, image_parts

    def _convert_image_block(self, block: dict, resolve: bool) -> dict | None:
        """image_ref → inline data URL for the current turn, else a placeholder."""
        btype = block.get("type")
        if btype == "image_ref":
            path = block.get("path")
            if resolve and path and os.path.exists(path):
                mime = block.get("mime_type") or "image/png"
                try:
                    mtime_ns = int(os.stat(path).st_mtime_ns)
                except OSError:
                    mtime_ns = 0
                data = _read_base64(path, mtime_ns)
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                }
            size_kb = int(block.get("size", 0)) // 1024
            return {
                "type": "text",
                "text": f"[图片 id={block.get('id', '?')} "
                        f"({block.get('mime_type', '?')} {size_kb}KB)；如需再查看请调用 attach_image]",
            }
        if btype == "image":
            mime = block.get("mimeType") or "image/png"
            data = block.get("data")
            if data:  # legacy inline block — already in memory, always resolve
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                }
        return None


DEFAULT_MODEL = Model(
    id="deepseek-chat",
    name="",
    provider="",
    base_url="https://api.deepseek.com",
    api="openai-compatible",
    reasoning=True,
    context_window=65536,
    max_tokens=8192,
)
