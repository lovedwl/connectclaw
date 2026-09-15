"""attach_image tool — re-attach a previously sent image into the current turn.

Image attach policy is "use-then-drop": images live in the session only as
lightweight image_ref blocks (path + mime + size, KB-level) and are resolved
to inline image_url only for the current turn. When the model needs to look
at an image from history again, it calls this tool; the returned image_ref
rides in the tool result and the provider resolves it for this turn.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.logging import get_logger

logger = get_logger(__name__)


class AttachmentStore:
    """In-process registry of downloaded images, keyed by short image id.

    main.py registers each downloaded image here; attach_image looks ids up.
    Persisted alongside the files in manifest.json so ids survive restarts.
    """

    def __init__(self, attachments_dir: str):
        self._dir = attachments_dir
        self._items: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._load_manifest()

    # ── Manifest persistence ─────────────────────────────────

    @property
    def _manifest_path(self) -> str:
        return os.path.join(self._dir, "manifest.json")

    def _load_manifest(self) -> None:
        try:
            import json

            with open(self._manifest_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, dict):
                self._items = {
                    k: v for k, v in items.items()
                    if isinstance(v, dict) and os.path.isfile(v.get("path", ""))
                }
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("attachments manifest load failed: %s", e)

    def _save_manifest(self) -> None:
        import json

        os.makedirs(self._dir, exist_ok=True)
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._items, f, ensure_ascii=False, indent=1)

    # ── Registry API ─────────────────────────────────────────

    async def register(
        self, image_id: str, path: str, mime_type: str, size: int
    ) -> dict:
        """Register a downloaded image (id -> image_ref dict)."""
        async with self._lock:
            ref = {
                "id": image_id,
                "path": path,
                "mime_type": mime_type,
                "size": size,
            }
            self._items[image_id] = ref
            try:
                self._save_manifest()
            except Exception as e:
                logger.warning("attachments manifest save failed: %s", e)
            return ref

    async def get(self, image_id: str) -> dict | None:
        async with self._lock:
            ref = self._items.get(image_id)
            if ref and not os.path.isfile(ref.get("path", "")):
                return None
            return ref

    async def list_ids(self) -> list[dict]:
        async with self._lock:
            return [
                {**ref, "exists": os.path.isfile(ref.get("path", ""))}
                for ref in self._items.values()
            ]


def make_image_ref(item: dict) -> dict:
    """Build an image_ref content block from a store entry."""
    return {
        "type": "image_ref",
        "id": item["id"],
        "path": item["path"],
        "mime_type": item["mime_type"],
        "size": item["size"],
    }


class AttachImageTool(AgentTool):
    name = "attach_image"
    label = "attach_image"
    description = (
        "Re-attach an image that was sent earlier in this conversation into "
        "the current turn so you can look at it again. Pass the image id "
        "shown next to the image (e.g. from '[image id=a1b2 ...]'). "
        "Use list_images=true to see all available images."
    )
    parameters = {
        "type": "object",
        "properties": {
            "image_id": {
                "type": "string",
                "description": "Id of the image to attach (from the [image id=...] marker)",
            },
            "list_images": {
                "type": "boolean",
                "description": "If true, list all available images instead of attaching",
            },
        },
        "required": [],
    }

    def __init__(self, store: AttachmentStore):
        self._store = store

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        if params.get("list_images"):
            items = await self._store.list_ids()
            if not items:
                return AgentToolResult(
                    content=[{"type": "text", "text": "没有可用的图片。"}],
                )
            lines = [
                f"- id={it['id']} {it['mime_type']} {it['size'] // 1024}KB"
                + ("" if it["exists"] else " (文件已丢失)")
                for it in items
            ]
            return AgentToolResult(
                content=[{"type": "text", "text": "可用图片：\n" + "\n".join(lines)}],
            )

        image_id = (params.get("image_id") or "").strip()
        if not image_id:
            return AgentToolResult(
                content=[{"type": "text", "text": "Error: 需要 image_id 参数（或 list_images=true）。"}],
                details={"is_error": True},
            )

        item = await self._store.get(image_id)
        if item is None:
            return AgentToolResult(
                content=[{"type": "text",
                          "text": f"Error: 找不到图片 id={image_id}。可用 list_images=true 查看全部。"}],
                details={"is_error": True},
            )

        logger.debug("attach_image: attaching %s (%s)", image_id, item["path"])
        return AgentToolResult(
            content=[
                {"type": "text", "text": f"图片 id={image_id} 已附加到本轮上下文。"},
                make_image_ref(item),
            ],
            details={"image_id": image_id, "path": item["path"]},
        )


def create_attach_image_tool(attachments_dir: str) -> AttachImageTool:
    return AttachImageTool(AttachmentStore(attachments_dir))
