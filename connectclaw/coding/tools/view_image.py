"""view_image tool — show a local image file to the multimodal model.

Ported from codex-rs `core/src/tools/handlers/view_image.rs` (Apache-2.0):
1. Resolve ``path`` against the agent cwd (same contract as read);
2. Validate the bytes really are an image (magic-number check, no new deps —
   codex uses the full ``image`` crate decoder for the same purpose);
3. Return an ``image_ref`` block in the tool result. The provider resolves it
   to an inline ``image_url`` for the current turn (same mechanism as
   attach_image), and older turns degrade to a short placeholder so the
   base64 never pollutes the prefix cache.

背景（2026-10-16 表情包事故）：image_analyze 退役后模型没有任何"主动看
本地图片"的工具——用户发来的图靠多模态内联，本地文件无路可走，模型凭
记忆找死工具 + bash 手搓 vision 无限挣扎。codex 的答案是一个独立、能力
边界清晰的 view_image 工具。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import aiofiles

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.logging import get_logger
from connectclaw.security import is_protected, path_within_cwd, resolve_tool_path

logger = get_logger(__name__)

# codex 同款语义：超限/非图片直接拒绝。20MB 与主流网关的图片输入上限对齐。
_MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Magic-number → mime（不引入 Pillow，够用的验证层）
_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
]

_VIEW_IMAGE_INVALID = "无法处理图片：无效或不受支持的图片数据"


def _sniff_mime(head: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if head.startswith(magic):
            return mime
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def _sniff_svg(sample: bytes) -> bool:
    return b"<svg" in sample[:1024].lstrip().lower() or b"<?xml" in sample[:256].lower()


class ViewImageTool(AgentTool):
    name = "view_image"
    label = "View Image"
    description = (
        "Show a local image file (png/jpeg/gif/webp/bmp/tiff/svg, max 20MB, "
        "within the working directory) to you as an inline image. "
        "Use when asked to look at, check, or pick image files on disk. "
        "Chat history images: attach_image; send files: send_file."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the image file (relative to cwd, or absolute within cwd)",
            },
            "detail": {
                "type": "string",
                "enum": ["high", "low", "auto"],
                "description": (
                    "Optional image processing fidelity hint for the model gateway. "
                    "high = full resolution (more tokens); low = downscaled (fewer "
                    "tokens); auto (default) = gateway decides."
                ),
            },
        },
        "required": ["path"],
    }

    def __init__(self, cwd: str, allow_outside_cwd: bool = False):
        self._cwd = cwd
        self._allow_outside_cwd = allow_outside_cwd
        if allow_outside_cwd:
            self.description = (
                "Show a local image file (png/jpeg/gif/webp/bmp/tiff/svg, max 20MB, "
                "any path on the host — full host access is enabled) to you as an "
                "inline image. Use when asked to look at, check, or pick image files "
                "on disk. Chat history images: attach_image; send files: send_file."
            )

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        file_path = (params.get("path") or "").strip()
        if not file_path:
            return AgentToolResult(
                content=[{"type": "text", "text": "错误：缺少必填参数 path。"}],
                details={"is_error": True},
            )

        absolute_path = resolve_tool_path(self._cwd, file_path)

        # 与 read 同一契约（H2）：默认 cwd 容器（sandbox=none 时解除）+ 逃生配置拒绝
        if not self._allow_outside_cwd and not path_within_cwd(self._cwd, absolute_path):
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"错误：只能查看工作目录（{self._cwd}）内的图片。"
                        f"'{file_path}' 在目录之外——可先用 bash 把它复制进工作目录。"
                    ),
                }],
                details={"is_error": True},
            )
        if is_protected(absolute_path):
            return AgentToolResult(
                content=[{"type": "text",
                          "text": f"⛔ 受保护文件（逃生配置，agent 不可读取）：{absolute_path}"}],
                details={"is_error": True},
            )

        if not os.path.isfile(absolute_path):
            return AgentToolResult(
                content=[{"type": "text", "text": f"错误：图片不存在：{file_path}"}],
                details={"is_error": True},
            )

        # codex 同款：先验大小，再验字节真的是图片
        try:
            size = os.path.getsize(absolute_path)
        except OSError as e:
            return AgentToolResult(
                content=[{"type": "text", "text": f"读取图片错误：{e}"}],
                details={"is_error": True},
            )
        if size > _MAX_IMAGE_BYTES:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"错误：图片过大（{size // (1024 * 1024)}MB > "
                        f"{_MAX_IMAGE_BYTES // (1024 * 1024)}MB 上限）。"
                        f"可先用 bash（convert -resize）缩放。"
                    ),
                }],
                details={"is_error": True},
            )
        if size == 0:
            return AgentToolResult(
                content=[{"type": "text", "text": f"错误：文件为空：{file_path}"}],
                details={"is_error": True},
            )

        try:
            async with aiofiles.open(absolute_path, "rb") as f:
                head = await f.read(1024)
                await f.seek(0)
                data = await f.read()
        except Exception as e:
            return AgentToolResult(
                content=[{"type": "text", "text": f"读取图片错误：{e}"}],
                details={"is_error": True},
            )

        mime = _sniff_mime(head)
        if mime is None and _sniff_svg(head):
            mime = "image/svg+xml"
        if mime is None:
            # codex: "unable to process image: invalid or unsupported image data"
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"{_VIEW_IMAGE_INVALID}：{file_path}。"
                        f"可用 bash（file/identify）确认文件类型。"
                    ),
                }],
                details={"is_error": True},
            )

        image_ref = {
            "type": "image_ref",
            "id": f"view-{abs(hash(absolute_path)) % 10**12:012d}",
            "path": absolute_path,
            "mime_type": mime,
            "size": size,
        }
        detail = (params.get("detail") or "").strip().lower()
        if detail in ("high", "low", "auto"):
            image_ref["detail"] = detail
        logger.info("view_image: %s (%s, %dKB)", absolute_path, mime, size // 1024)
        return AgentToolResult(
            content=[
                {
                    "type": "text",
                    "text": f"已加载图片 {file_path}（{mime}, {size // 1024}KB）——见随附图像。",
                },
                image_ref,
            ],
        )


def create_view_image_tool(cwd: str, allow_outside_cwd: bool = False) -> ViewImageTool:
    return ViewImageTool(cwd, allow_outside_cwd=allow_outside_cwd)
