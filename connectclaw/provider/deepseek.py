"""OpenAI-compatible LLM provider."""

import json

from openai import AsyncOpenAI

from .types import (
    AssistantMessage,
    Model,
    StreamEvent,
    ToolDef,
)


class DeepSeekProvider:
    """OpenAI-compatible provider with reasoning_content support."""

    def __init__(self, base_url: str = "https://api.deepseek.com"):
        self.base_url = base_url

    def build_client(self, api_key: str, base_url: str | None = None) -> AsyncOpenAI:
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
        """Convert ConnectClaw messages to OpenAI format."""
        result = []
        for m in messages:
            role = m.role
            if role == "user":
                result.append(self._convert_user_message(m))
            elif role == "assistant":
                result.append(self._convert_assistant_message(m))
            elif role == "toolResult":
                result.append(self._convert_tool_result(m))
        return result

    def _convert_user_message(self, m) -> dict:
        content = m.content
        if isinstance(content, str):
            return {"role": "user", "content": content}
        # Content is a list of blocks
        parts = []
        for block in content:
            block_type = block.get("type", block["type"])
            if block_type == "text":
                parts.append({"type": "text", "text": block.get("text", block["text"])})
            elif block_type == "image":
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block.get('mimeType', block['mimeType'])};base64,{block.get('data', block['data'])}"
                    },
                })
        return {"role": "user", "content": parts}

    def _convert_assistant_message(self, m) -> dict:
        content = m.content
        msg: dict = {"role": "assistant", "content": ""}
        text_parts = []
        tool_calls = []
        thinking_parts = []

        for block in content:
            block_type = block.get("type", block["type"])
            if block_type == "text":
                text_parts.append(block.get("text", block["text"]))
            elif block_type == "thinking":
                thinking_text = block.get("thinking", block["thinking"])
                # reasoning_content carries thinking output
                thinking_parts.append(thinking_text)
            elif block_type == "toolCall":
                tool_calls.append({
                    "id": block.get("id", block["id"]),
                    "type": "function",
                    "function": {
                        "name": block.get("name", block["name"]),
                        "arguments": json.dumps(block.get("arguments", block["arguments"])),
                    },
                })

        msg["content"] = "\n".join(text_parts) or None
        if thinking_parts:
            msg["reasoning_content"] = "\n".join(thinking_parts)
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    def _convert_tool_result(self, m) -> dict:
        content = m.content
        text = ""
        for block in content:
            if block.get("type", block["type"]) == "text":
                text += block.get("text", block["text"])
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "content": text,
        }

    def map_chunk_to_event(
        self, chunk, model: Model, partial: AssistantMessage, content_index: int
    ) -> StreamEvent | None:
        """Map an OpenAI SSE chunk to a StreamEvent."""
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            return None

        # Reasoning / thinking content (DeepSeek-specific)
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning = delta.reasoning_content
            if partial.content and partial.content[-1].get("type") != "thinking":
                partial.content.append({"type": "thinking", "thinking": ""})
            idx = len(partial.content) - 1
            partial.content[idx]["thinking"] += reasoning
            return StreamEvent(
                type="thinking_delta",
                delta=reasoning,
                content_index=idx,
                partial=partial,
            )

        # Regular text content
        if delta.content:
            text = delta.content
            if not partial.content or partial.content[-1].get("type") != "text":
                partial.content.append({"type": "text", "text": ""})
            idx = len(partial.content) - 1
            partial.content[idx]["text"] += text
            return StreamEvent(
                type="text_delta",
                delta=text,
                content_index=idx,
                partial=partial,
            )

        # Tool calls
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                # Ensure the content slot exists
                while len(partial.content) <= idx:
                    partial.content.append({"type": "toolCall", "id": "", "name": "", "arguments": {}})
                existing = partial.content[idx]
                if tc_delta.id:
                    existing["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        existing["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        existing["arguments"] = json.loads(tc_delta.function.arguments)
                partial.content[idx] = dict(existing)
                return StreamEvent(
                    type="toolcall_delta",
                    delta=json.dumps(existing.get("arguments", {})),
                    content_index=idx,
                    partial=partial,
                )

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
