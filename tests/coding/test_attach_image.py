"""attach_image tool + AttachmentStore tests.

Covers the use-then-drop image policy infrastructure:
  1. AttachmentStore registers / persists / loads a manifest of image_refs.
  2. attach_image re-attaches a registered image and lists available ones.
  3. Missing ids and files produce clean errors instead of crashes.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from connectclaw.coding.tools.attach_image import (
    AttachImageTool,
    AttachmentStore,
    make_image_ref,
)


@pytest.fixture
def attachments_dir(tmp_path) -> str:
    return str(tmp_path / "attachments")


@pytest.fixture
def sample_image(attachments_dir) -> str:
    path = os.path.join(attachments_dir, "sample.png")
    os.makedirs(attachments_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x89PNG fake bytes")
    return path


# ── AttachmentStore ────────────────────────────────────────────


def test_store_register_roundtrip(attachments_dir, sample_image):
    store = AttachmentStore(attachments_dir)
    ref = asyncio.run(store.register("id1", sample_image, "image/png", 14))
    assert ref == {"id": "id1", "path": sample_image, "mime_type": "image/png", "size": 14}
    got = asyncio.run(store.get("id1"))
    assert got and got["id"] == "id1"
    assert asyncio.run(store.get("nope")) is None


def test_store_manifest_survives_reload(attachments_dir, sample_image):
    asyncio.run(AttachmentStore(attachments_dir).register("id1", sample_image, "image/png", 14))
    store2 = AttachmentStore(attachments_dir)
    assert asyncio.run(store2.get("id1")) is not None


def test_store_drops_entry_for_missing_file(attachments_dir):
    store = AttachmentStore(attachments_dir)
    # Register an entry then delete the file — get() must return None, and a
    # store reload must not resurrect a dangling entry.
    path = os.path.join(attachments_dir, "zzz.png")
    os.makedirs(attachments_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")
    asyncio.run(store.register("zzz", path, "image/png", 1))
    os.remove(path)
    assert asyncio.run(store.get("zzz")) is None
    store2 = AttachmentStore(attachments_dir)
    assert asyncio.run(store2.get("zzz")) is None


def test_store_evicts_oldest_over_caps(attachments_dir, monkeypatch):
    """超过数量/字节上限时按注册先后淘汰最旧的（文件与条目一起删）。"""
    import connectclaw.coding.tools.attach_image as ai

    monkeypatch.setattr(ai, "_MAX_ATTACHMENTS", 2)
    monkeypatch.setattr(ai, "_MAX_ATTACHMENT_BYTES", 10 * 1024 * 1024)
    store = AttachmentStore(attachments_dir)

    os.makedirs(attachments_dir, exist_ok=True)
    paths = []
    for i in range(3):
        p = os.path.join(attachments_dir, f"img{i}.png")
        with open(p, "wb") as f:
            f.write(f"img{i}".encode() * 1000)
        asyncio.run(store.register(f"id{i}", p, "image/png", os.path.getsize(p)))
        paths.append(p)

    # 最旧的 id0 被数量上限淘汰；id1/id2 保留。
    assert asyncio.run(store.get("id0")) is None
    assert asyncio.run(store.get("id1")) is not None
    assert asyncio.run(store.get("id2")) is not None
    assert not os.path.exists(paths[0])
    assert os.path.exists(paths[1]) and os.path.exists(paths[2])


def test_make_image_ref():
    item = {"id": "a1", "path": "/x/y.png", "mime_type": "image/png", "size": 2048}
    assert make_image_ref(item) == {
        "type": "image_ref",
        "id": "a1",
        "path": "/x/y.png",
        "mime_type": "image/png",
        "size": 2048,
    }


# ── AttachImageTool ────────────────────────────────────────────


async def _run_execute(store: AttachmentStore, params: dict) -> tuple[list, Any]:
    tool = AttachImageTool(store)
    return await tool.execute("tc1", params)


async def test_attach_returns_image_ref_with_text(attachments_dir, sample_image):
    store = AttachmentStore(attachments_dir)
    await store.register("abc", sample_image, "image/png", 14)
    tool = AttachImageTool(store)
    result = await tool.execute("tc1", {"image_id": "abc"})
    types = [b["type"] for b in result.content]
    assert types == ["text", "image_ref"]
    ref = result.content[1]
    assert ref["id"] == "abc" and ref["path"] == sample_image


async def test_attach_missing_id_errors(attachments_dir):
    store = AttachmentStore(attachments_dir)
    tool = AttachImageTool(store)
    result = await tool.execute("tc1", {"image_id": "ghost"})
    assert result.details and result.details.get("is_error")
    assert "找不到" in result.content[0]["text"]


async def test_list_images(attachments_dir, sample_image):
    store = AttachmentStore(attachments_dir)
    await store.register("one", sample_image, "image/png", 2048)
    tool = AttachImageTool(store)
    result = await tool.execute("tc1", {"list_images": True})
    text = result.content[0]["text"]
    assert "one" in text and "image/png" in text and "2KB" in text
