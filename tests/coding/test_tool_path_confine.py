"""2026-10 (H2/M24) regression tests — file tools are confined to cwd and
must never touch the escape-hatch config.

Covers:
- read / hash_read / hash_edit / write refuse paths outside cwd
- absolute paths containing `..` cannot bypass the cwd prefix check
- is_protected resolves symlinks (realpath), not just normpath
"""

from __future__ import annotations

import os

import pytest

from connectclaw.coding.tools.hash_edit import create_hash_edit_tool
from connectclaw.coding.tools.hash_read import create_hash_read_tool
from connectclaw.coding.tools.read import create_read_tool
from connectclaw.coding.tools.write import create_write_tool
from connectclaw.security import is_protected, resolve_tool_path


@pytest.fixture()
def layout(tmp_path):
    cwd = tmp_path / "proj"
    outside = tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    return str(cwd), str(outside)


async def test_read_refuses_outside_cwd(layout):
    cwd, outside = layout
    victim = os.path.join(outside, "secret.txt")
    with open(victim, "w") as f:
        f.write("top secret\n")
    rt = create_read_tool(cwd)
    r = await rt.execute("t", {"file_path": victim})
    assert r.details.get("is_error") is True
    assert "目录之外" in r.content[0]["text"]
    assert "top secret" not in r.content[0]["text"]


async def test_read_refuses_dotdot_escape(layout):
    cwd, outside = layout
    victim = os.path.join(outside, "secret.txt")
    with open(victim, "w") as f:
        f.write("top secret\n")
    rt = create_read_tool(cwd)
    r = await rt.execute("t", {"file_path": os.path.join(cwd, "..", "outside", "secret.txt")})
    assert r.details.get("is_error") is True
    assert "top secret" not in r.content[0]["text"]


async def test_hash_read_refuses_outside_cwd(layout):
    cwd, outside = layout
    victim = os.path.join(outside, "config.toml")
    with open(victim, "w") as f:
        f.write("app_secret = 'x'\n")
    hr = create_hash_read_tool(cwd)
    r = await hr.execute("t", {"path": victim})
    assert r.details.get("is_error") is True
    assert "app_secret" not in r.content[0]["text"]


async def test_hash_edit_refuses_outside_cwd(layout):
    cwd, outside = layout
    victim = os.path.join(outside, "authorized_keys")
    with open(victim, "w") as f:
        f.write("ssh-ed25519 AAAA\n")
    he = create_hash_edit_tool(cwd)
    r = await he.execute("t", {
        "path": victim,
        "edits": [{"op": "append", "lines": ["evil"]}],
    })
    assert r.details.get("is_error") is True
    assert "目录之外" in r.content[0]["text"]
    with open(victim) as f:
        assert "evil" not in f.read()


async def test_write_dotdot_prefix_cannot_bypass_cwd_check(layout):
    cwd, outside = layout
    # cwd = /tmp/xxx/proj → /tmp/xxx/proj/../outside/f must not pass the
    # startswith("/…/proj/") prefix check (M24).
    tricky = os.path.join(cwd, "..", "outside", "f.txt")
    wt = create_write_tool(cwd, read_tool=None)
    r = await wt.execute("t", {"file_path": tricky, "content": "escaped\n"})
    assert r.details.get("is_error") is True
    assert not os.path.exists(os.path.join(outside, "f.txt"))


async def test_is_protected_resolves_symlinks(tmp_path, monkeypatch):
    # ln -s <real config> inside cwd → edit via the link must still be refused
    real_cfg = tmp_path / "real" / "config.toml"
    real_cfg.parent.mkdir()
    real_cfg.write_text("[bash]\n")
    monkeypatch.setattr(
        "connectclaw.security.protected_file_paths",
        lambda: [str(real_cfg)],
    )
    link = tmp_path / "link"
    os.symlink(str(real_cfg.parent), str(link))
    assert is_protected(str(link / "config.toml")) is True
    assert is_protected(str(tmp_path / "other.toml")) is False


def test_resolve_tool_path_normalizes_absolute_dotdot():
    p = resolve_tool_path("/proj", "/proj/../proj2/f.txt")
    assert p == "/proj2/f.txt"
    p = resolve_tool_path("/proj", "../proj2/f.txt")
    assert p == "/proj2/f.txt"


# ── P2：read 读图片/二进制的明确指引 ──────────────────────────


async def test_read_image_extension_gives_clear_guidance(layout):
    from connectclaw.coding.tools.read import create_read_tool as _crt

    cwd, _ = layout
    img = os.path.join(cwd, "sticker.gif")
    with open(img, "wb") as f:
        f.write(b"GIF89a" + b"\x00" * 32)
    rt = _crt(cwd)
    r = await rt.execute("t", {"file_path": img})
    assert r.details.get("is_error") is True
    text = r.content[0]["text"]
    assert "图片" in text
    assert "send_file" in text  # 必须指路正确工具
    # 不能是晦涩的 codec 报错
    assert "codec" not in text


async def test_read_binary_file_gives_clear_guidance(layout):
    from connectclaw.coding.tools.read import create_read_tool as _crt

    cwd, _ = layout
    blob = os.path.join(cwd, "blob.dat")
    with open(blob, "wb") as f:
        f.write(bytes(range(256)) * 8)
    rt = _crt(cwd)
    r = await rt.execute("t", {"file_path": blob})
    assert r.details.get("is_error") is True
    text = r.content[0]["text"]
    assert "二进制" in text
    assert "send_file" in text
    assert "codec" not in text
