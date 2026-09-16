"""Media send wire-format tests — the exact SDK payload for images/files.

Regression for the "send 图片失败（无 message_id）" bug: the SDK's send()
takes ``{"image": {"source": <path>}}`` / ``{"file": {"source": <path>,
"file_name": ...}}`` and uploads+sends in one step — NOT upload_media +
image_key/file_key. Pinned here so a refactor can't silently reintroduce the
wrong payload.
"""

from __future__ import annotations

from types import SimpleNamespace

from connectclaw.channel.feishu import FeishuChannel
from connectclaw.config import FeishuConfig


class FakeSdk:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    async def send(self, chat: str, payload: dict):
        self.sent.append((chat, payload))
        return SimpleNamespace(ok=True, message_id="mid-1", error=None)


def _channel_with_sdk() -> tuple[FeishuChannel, FakeSdk]:
    ch = FeishuChannel(FeishuConfig(app_id="a", app_secret="b"))
    sdk = FakeSdk()
    ch._sdk = sdk
    return ch, sdk


async def test_send_image_payload_is_source_based():
    ch, sdk = _channel_with_sdk()
    mid = await ch.send_image("oc_x", "/home/rolia/.connectclaw/me.png")
    assert mid == "mid-1"
    assert sdk.sent == [
        ("oc_x", {"image": {"source": "/home/rolia/.connectclaw/me.png"}})
    ]


async def test_send_file_payload_is_source_based_with_name():
    ch, sdk = _channel_with_sdk()
    mid = await ch.send_file("oc_x", "/home/rolia/.connectclaw/report.md")
    assert mid == "mid-1"
    assert sdk.sent == [
        ("oc_x", {"file": {"source": "/home/rolia/.connectclaw/report.md",
                           "file_name": "report.md"}})
    ]


async def test_send_image_not_connected_returns_empty():
    ch = FeishuChannel(FeishuConfig(app_id="a", app_secret="b"))
    assert await ch.send_image("oc_x", "/tmp/a.png") == ""
    assert await ch.send_file("oc_x", "/tmp/a.txt") == ""
