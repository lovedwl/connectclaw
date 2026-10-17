"""2026-10 (view_image) regression tests — local-image viewing, codex-style.

Covers:
- a real image file returns an image_ref block the provider can inline
- non-image bytes are rejected (codex: "invalid or unsupported image data")
- path contract: cwd containment, same as read
- oversize images are refused
"""

from __future__ import annotations

import base64
import os

import pytest

from connectclaw.coding.tools.view_image import create_view_image_tool

# 1x1 transparent PNG
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def env(tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    return str(cwd)


async def test_view_image_returns_image_ref(env):
    img = os.path.join(env, "sticker.png")
    with open(img, "wb") as f:
        f.write(_TINY_PNG)
    tool = create_view_image_tool(env)
    r = await tool.execute("t", {"path": "sticker.png"})
    assert (r.details or {}).get("is_error") is not True
    refs = [b for b in r.content if b.get("type") == "image_ref"]
    assert len(refs) == 1
    assert refs[0]["path"] == img
    assert refs[0]["mime_type"] == "image/png"
    assert refs[0]["size"] == len(_TINY_PNG)


async def test_view_image_rejects_non_image(env):
    f = os.path.join(env, "not-an-image.png")  # 扩展名骗不了魔数校验
    with open(f, "w") as fh:
        fh.write("arbitrary file contents")
    tool = create_view_image_tool(env)
    r = await tool.execute("t", {"path": f})
    assert r.details.get("is_error") is True
    assert "无效或不受支持的图片数据" in r.content[0]["text"]


async def test_view_image_rejects_outside_cwd(env, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    img = outside / "x.png"
    img.write_bytes(_TINY_PNG)
    tool = create_view_image_tool(env)
    r = await tool.execute("t", {"path": str(img)})
    assert r.details.get("is_error") is True
    assert "目录之外" in r.content[0]["text"]


async def test_view_image_rejects_oversize(env, monkeypatch):
    import connectclaw.coding.tools.view_image as vi

    img = os.path.join(env, "big.png")
    with open(img, "wb") as f:
        f.write(_TINY_PNG + b"\x00" * 64)
    monkeypatch.setattr(vi, "_MAX_IMAGE_BYTES", 32)
    tool = create_view_image_tool(env)
    r = await tool.execute("t", {"path": img})
    assert r.details.get("is_error") is True
    assert "过大" in r.content[0]["text"]


async def test_view_image_gif_and_webp_magic(env):
    tool = create_view_image_tool(env)
    gif = os.path.join(env, "a.gif")
    with open(gif, "wb") as f:
        f.write(b"GIF89a" + b"\x00" * 16)
    r = await tool.execute("t", {"path": gif})
    refs = [b for b in r.content if b.get("type") == "image_ref"]
    assert refs and refs[0]["mime_type"] == "image/gif"

    webp = os.path.join(env, "b.webp")
    with open(webp, "wb") as f:
        f.write(b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 16)
    r = await tool.execute("t", {"path": webp})
    refs = [b for b in r.content if b.get("type") == "image_ref"]
    assert refs and refs[0]["mime_type"] == "image/webp"
