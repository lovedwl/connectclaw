"""DeepSeek provider image-resolution policy tests (use-then-drop).

The provider resolves image_ref blocks to inline data URLs ONLY for the current
turn (last user message and anything after it, e.g. attach_image tool results);
images in older history degrade to a short text placeholder so base64 never
lives in the long-lived, prefix-cached text prefix.
"""

from __future__ import annotations

import os

from connectclaw.provider.deepseek import DeepSeekProvider
from connectclaw.provider.types import UserMessage, ToolResultMessage

provider = DeepSeekProvider()


def _img_ref(image_id: str, path: str, size: int = 2048) -> dict:
    return {"type": "image_ref", "id": image_id, "path": path,
            "mime_type": "image/png", "size": size}


def _mk_user(*blocks: dict) -> UserMessage:
    return UserMessage(
        content=[{"type": "text", "text": "turn text"}, *blocks],
        timestamp=0,
    )


def test_current_turn_image_resolves_to_data_url(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\x0d\x0a\x1a\x0a")

    msgs = [_mk_user(_img_ref("a1", str(img)))]
    out = provider.convert_messages(msgs)

    assert out[0]["role"] == "user"
    content = out[0]["content"]
    kinds = [b["type"] for b in content]
    assert "image_url" in kinds
    url = next(b["image_url"]["url"] for b in content if b["type"] == "image_url")
    assert url.startswith("data:image/png;base64,")


def test_historic_image_degrades_to_placeholder(tmp_path):
    old = tmp_path / "old.png"
    old.write_bytes(b"x")

    # First turn carries the image; the SECOND (later) turn is the current one,
    # so the first turn's image is history → placeholder, not a data URL.
    msgs = [
        _mk_user(_img_ref("a1", str(old))),
        _mk_user(),
    ]
    out = provider.convert_messages(msgs)
    first = out[0]["content"]
    assert "image_url" not in [blk.get("type") for blk in first]
    text = " ".join(blk.get("text", "") for blk in first)
    assert "a1" in text and "attach_image" in text


def test_missing_file_yields_placeholder_even_current_turn(tmp_path):
    msgs = [_mk_user(_img_ref("ghost", str(tmp_path / "nope.png")))]
    out = provider.convert_messages(msgs)
    text = " ".join(b.get("text", "") for b in out[0]["content"])
    assert "ghost" in text and "attach_image" in text


def test_only_last_user_turn_resolves(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")

    msgs = [
        _mk_user(_img_ref("old1", str(a))),
        _mk_user(_img_ref("new1", str(b))),
    ]
    out = provider.convert_messages(msgs)
    assert len(out) == 2

    first = out[0]["content"]
    assert "image_url" not in [blk.get("type") for blk in first]
    assert any("old1" in blk.get("text", "") for blk in first)

    second = out[1]["content"]
    assert "image_url" in [blk.get("type") for blk in second]


def test_attach_image_tool_result_rides_synthetic_user_message(tmp_path):
    """toolResult 里的 image_ref 转换为独立的 user 消息（tool 消息保持纯文本）。

    真实顺序：user 消息 → assistant tool_call → toolResult。最后一条 user 消息
    之后的内容（含 attach_image 的 toolResult）都按当前轮处理、解析图片。
    """
    img = tmp_path / "c.png"
    img.write_bytes(b"ccc")

    current = UserMessage(content=[{"type": "text", "text": "show me"}], timestamp=0)
    tool_result = ToolResultMessage(
        tool_call_id="tc1",
        content=[{"type": "text", "text": "attached"}, _img_ref("c1", str(img))],
        timestamp=0,
    )

    out = provider.convert_messages([current, tool_result])

    roles = [m["role"] for m in out]
    assert roles[-1] == "user"

    # The synthetic user message after the tool result carries the resolved image.
    last = out[-1]
    kinds = [b["type"] for b in last["content"]]
    assert "image_url" in kinds

    # The tool message itself stays text-only.
    tool_msg = next(m for m in out if m["role"] == "tool")
    assert "image_url" not in str(tool_msg)


def test_historic_tool_result_keeps_placeholder(tmp_path):
    """历史 toolResult 里的 image_ref 降级后必须保留文字占位符——否则模型会
    看到“已附加”，既看不到图，也没有“可调用 attach_image”的提示。"""
    old = tmp_path / "old.png"
    old.write_bytes(b"x")

    first = UserMessage(content=[{"type": "text", "text": "first"}], timestamp=0)
    tool_result = ToolResultMessage(
        tool_call_id="tc1",
        content=[{"type": "text", "text": "attached"}, _img_ref("c1", str(old))],
        timestamp=0,
    )
    later = UserMessage(content=[{"type": "text", "text": "later turn"}], timestamp=0)

    out = provider.convert_messages([first, tool_result, later])

    tool_msg = next(m for m in out if m["role"] == "tool")
    text = tool_msg["content"]
    assert "attached" in text
    assert "c1" in text and "attach_image" in text


def test_legacy_image_block_resolves_without_file(tmp_path):
    """旧式 {'type':'image','data':...} 块始终解析（数据已在内存中）。"""
    import base64
    from connectclaw.provider.types import UserMessage

    b64 = base64.b64encode(b"legacy-data").decode()
    msgs = [UserMessage(
        content=[{"type": "image", "mimeType": "image/jpeg", "data": b64}],
        timestamp=0,
    )]
    out = provider.convert_messages(msgs)
    kinds = [b["type"] for b in out[0]["content"]]
    assert "image_url" in kinds
