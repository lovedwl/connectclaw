"""attach_image tool — re-attach a previously sent image into the current turn.

Image attach policy is "use-then-drop": images live in the session only as
lightweight image_ref blocks (path + mime + size, KB-level) and are resolved
to inline image_url only for the current turn. When the model needs to look
at an image from history again, it calls this tool; the returned image_ref
rides in the tool result and the provider resolves it for this turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.logging import get_logger

logger = get_logger(__name__)

# Shared location for downloaded Feishu images. A single constant so main.py
# (which writes the files) and coding_agent.py (which owns the store) can't
# drift apart.
DEFAULT_ATTACHMENTS_DIR = os.path.join(os.path.expanduser("~/.connectclaw"), "attachments")

# The attachments dir + manifest are bounded: at most _MAX_ATTACHMENTS files
# and _MAX_ATTACHMENT_BYTES total, oldest evicted first when a register pushes
# past either cap.
_MAX_ATTACHMENTS = 256
_MAX_ATTACHMENT_BYTES = 256 * 1024 * 1024

# Manifest writes are debounced: the first register in a burst persists
# immediately (an entry must never be lost to a crash or a prompt restart),
# later ones inside the window coalesce into one write after this delay.
_SAVE_DEBOUNCE_SECS = 2.0


class AttachmentStore:
    """In-process registry of downloaded images, keyed by short image id.

    main.py registers each downloaded image here; attach_image looks ids up.
    Persisted alongside the files in manifest.json so ids survive restarts.
    """

    def __init__(self, attachments_dir: str):
        self._dir = attachments_dir
        self._items: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._save_task: asyncio.Task | None = None
        self._last_saved_ts = 0.0
        self._load_manifest()

    # ── Manifest persistence ─────────────────────────────────

    @property
    def _manifest_path(self) -> str:
        return os.path.join(self._dir, "manifest.json")

    def _load_manifest(self) -> None:
        try:
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

    def _schedule_save(self) -> None:
        """Persist dirty state: immediately on the first write of a burst or
        when no event loop is running, otherwise coalesced into one debounced
        async write."""
        self._dirty = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None or self._last_saved_ts == 0 or (time.time() - self._last_saved_ts) > _SAVE_DEBOUNCE_SECS:
            self._flush_sync()
        elif self._save_task is None:
            self._save_task = loop.create_task(self._delayed_save())

    async def _delayed_save(self) -> None:
        try:
            await asyncio.sleep(_SAVE_DEBOUNCE_SECS)
            self._flush_sync()
        finally:
            self._save_task = None

    def _flush_sync(self) -> None:
        """Write the manifest from a snapshot. Safe without the lock: the event
        loop is single-threaded and register() mutates before any await."""
        if not self._dirty:
            return
        snapshot = dict(self._items)
        try:
            os.makedirs(self._dir, exist_ok=True)
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=1)
            self._dirty = False
            self._last_saved_ts = time.time()
        except Exception as e:
            logger.warning("attachments manifest save failed: %s", e)

    async def flush(self) -> None:
        """Persist any pending writes immediately (called on shutdown)."""
        task = self._save_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._save_task = None
        self._flush_sync()

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
            # ts drives oldest-first eviction; kept out of the returned ref so
            # image_ref blocks stay lean and stable.
            self._items[image_id] = {**ref, "ts": time.time()}
        await self._evict()
        self._schedule_save()
        return ref

    async def _evict(self) -> None:
        """Enforce the caps: drop the oldest entries (file + manifest entry)
        that push the store past the count or byte limits."""
        async with self._lock:
            total = sum(int(v.get("size", 0)) for v in self._items.values())
            if len(self._items) <= _MAX_ATTACHMENTS and total <= _MAX_ATTACHMENT_BYTES:
                return
            ordered = sorted(self._items, key=lambda k: self._items[k].get("ts", 0))
            dropped: list[dict] = []
            while (
                (len(self._items) > _MAX_ATTACHMENTS or total > _MAX_ATTACHMENT_BYTES)
                and ordered
            ):
                k = ordered.pop(0)
                entry = self._items.pop(k)
                dropped.append(entry)
                total -= int(entry.get("size", 0))
        for entry in dropped:
            try:
                os.remove(entry.get("path", ""))
            except OSError:
                pass
        if dropped:
            logger.info("attachments: evicted %d image(s) over cap (%d kept)",
                        len(dropped), len(self._items))

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
                content=[{"type": "text", "text": "错误：需要 image_id 参数（或 list_images=true）。"}],
                details={"is_error": True},
            )

        item = await self._store.get(image_id)
        if item is None:
            return AgentToolResult(
                content=[{"type": "text",
                          "text": f"错误：找不到图片 id={image_id}。可用 list_images=true 查看全部。"}],
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
