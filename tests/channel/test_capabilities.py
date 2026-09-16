"""Channel-capability tool tests — file/image delivery via the Channel ABC.

The tool (channel/capabilities.py) binds the abstract Channel only; a fake
channel verifies dispatch (image vs file, auto-detect, error paths).
"""

from __future__ import annotations

import os

from connectclaw.channel.capabilities import SendFileTool


class FakeChannel:
    def __init__(self) -> None:
        self.images: list[str] = []
        self.files: list[str] = []
        self.mid = "mid-ok"

    async def send_image(self, chat, path: str) -> str:
        self.images.append((chat, path))
        return self.mid

    async def send_file(self, chat, path: str) -> str:
        self.files.append((chat, path))
        return self.mid


def _tool(channel, chat="oc_test"):
    return SendFileTool(channel=channel, chat_provider=lambda: chat)


def _write(tmp_path, name: str) -> str:
    p = os.path.join(tmp_path, name)
    with open(p, "wb") as f:
        f.write(b"probe")
    return p


async def test_image_ext_sends_as_image(tmp_path):
    ch = FakeChannel()
    path = _write(tmp_path, "chart.png")
    tool = _tool(ch)
    r = await tool.execute("t1", {"path": path})
    assert ch.images == [("oc_test", path)]
    assert ch.files == []
    assert "图片" in r.content[0]["text"]
    assert r.details and r.details["message_id"] == "mid-ok"


async def test_other_ext_sends_as_file(tmp_path):
    ch = FakeChannel()
    path = _write(tmp_path, "report.md")
    tool = _tool(ch)
    r = await tool.execute("t1", {"path": path})
    assert ch.files == [("oc_test", path)]
    assert ch.images == []
    assert "文件" in r.content[0]["text"]


async def test_as_image_override(tmp_path):
    ch = FakeChannel()
    path = _write(tmp_path, "data.bin")
    tool = _tool(ch)
    r = await tool.execute("t1", {"path": path, "as_image": True})
    assert ch.images == [("oc_test", path)]
    assert ch.files == []


async def test_missing_file_errors(tmp_path):
    ch = FakeChannel()
    tool = _tool(ch)
    r = await tool.execute("t1", {"path": os.path.join(tmp_path, "nope.txt")})
    assert r.details and r.details.get("is_error")
    assert ch.images == [] and ch.files == []


async def test_no_active_chat_errors(tmp_path):
    ch = FakeChannel()
    path = _write(tmp_path, "x.txt")
    tool = SendFileTool(channel=ch, chat_provider=lambda: "")
    r = await tool.execute("t1", {"path": path})
    assert r.details and r.details.get("is_error")


async def test_send_failure_surfaces(tmp_path):
    class BoomChannel:
        async def send_file(self, chat, path):
            raise RuntimeError("upload rejected")

    path = _write(tmp_path, "x.txt")
    tool = _tool(BoomChannel())
    r = await tool.execute("t1", {"path": path})
    assert r.details and r.details.get("is_error")
    assert "upload rejected" in r.content[0]["text"]
