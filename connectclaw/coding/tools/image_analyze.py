"""
Image analyze tool — sub-agent that analyzes images using a vision model.

The sub-agent pattern:
1. Main agent calls image_analyze tool with an image path and question
2. Tool reads the image, calls vision model (OpenAI-compatible API)
3. Result is returned to main agent as tool result

This keeps image data out of the main agent's context window.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.provider.types import Context, Model


@dataclass
class VisionConfig:
    api_key: str = ""
    base_url: str = ""
    model_id: str = ""


class ImageAnalyzeTool(AgentTool):
    name = "image_analyze"
    label = "image_analyze"
    description = (
        "Analyze an image file using a vision model. "
        "Use this to understand images, screenshots, diagrams, or UI mockups. "
        "Returns a text description of the image content. "
        "The image is processed by a separate vision model, not the main agent."
    )
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Absolute path to the image file",
            },
            "question": {
                "type": "string",
                "description": "What to analyze in the image (default: 'Describe this image in detail')",
            },
        },
        "required": ["image_path"],
    }

    # Common image MIME types
    MIME_MAP = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __init__(
        self,
        config: VisionConfig,
        cwd: str = ".",
    ):
        self._config = config
        self._cwd = cwd

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        image_path = params["image_path"]
        question = params.get("question", "Describe this image in detail")

        # Resolve path
        if not os.path.isabs(image_path):
            image_path = os.path.normpath(os.path.join(self._cwd, image_path))

        # Check file exists
        if not os.path.isfile(image_path):
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": f"Error: Image file not found: {image_path}",
                }],
            )

        # Check file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = self.MIME_MAP.get(ext)
        if not mime_type:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": f"Error: Unsupported image format: {ext}. "
                            f"Supported: {', '.join(self.MIME_MAP.keys())}",
                }],
            )

        # If no API key, return placeholder
        if not self._config.api_key:
            try:
                file_size = os.path.getsize(image_path)
            except Exception:
                file_size = 0
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"Image analysis is not configured (VISION_API_KEY not set). "
                        f"Image: {image_path} ({file_size} bytes, {mime_type})\n"
                        f"Question: {question}\n\n"
                        f"To enable image analysis, set VISION_API_KEY in .env or config.toml."
                    ),
                }],
            )

        # Try to analyze via vision model
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Call vision model using OpenAI-compatible API
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url=self._config.base_url,
                api_key=self._config.api_key,
            )

            response = await client.chat.completions.create(
                model=self._config.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                            },
                        },
                    ],
                }],
                max_tokens=1000,
            )

            text = response.choices[0].message.content or "(no description)"

            return AgentToolResult(
                content=[{"type": "text", "text": text}],
                details={
                    "image_path": image_path,
                    "question": question,
                    "model": self._config.model_id,
                },
            )

        except ImportError:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"Image analysis requires the openai package. "
                        f"Image: {image_path}"
                    ),
                }],
            )
        except Exception as e:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": f"Image analysis failed: {e}",
                }],
            )


def create_image_analyze_tool(
    api_key: str = "",
    base_url: str = "",
    model_id: str = "",
    cwd: str = ".",
) -> ImageAnalyzeTool:
    return ImageAnalyzeTool(
        config=VisionConfig(
            api_key=api_key,
            base_url=base_url,
            model_id=model_id,
        ),
        cwd=cwd,
    )
