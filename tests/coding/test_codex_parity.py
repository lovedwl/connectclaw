"""2026-10-17 (codex-parity) tests — bash structured output, head/tail
truncation, view_image detail hint.

- bash toolResult is a Codex-style JSON object with fixed fields
  (output/exit_code/wall_time_s, optional stderr/timed_out/truncated)
- head_tail_truncate keeps head AND tail (error stacks live at the tail);
  stderr is truncated the same way
- view_image forwards the optional `detail` hint through image_ref, and the
  provider puts it into the wire-format image_url object
"""

from __future__ import annotations

import base64
import json

from connectclaw.coding.safety.sandbox import head_tail_truncate
from connectclaw.coding.tools.view_image import create_view_image_tool
from connectclaw.provider.deepseek import DeepSeekProvider

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


# ── head_tail_truncate ─────────────────────────────────────────


def test_head_tail_keeps_both_ends():
    text = "HEAD-MARKER\n" + "x" * 1000 + "\nTAIL-MARKER"
    out, truncated = head_tail_truncate(text, 200)
    assert truncated is True
    assert "HEAD-MARKER" in out   # 头部保留
    assert "TAIL-MARKER" in out   # 尾部保留（错误堆栈在尾部）
    assert "中间省略" in out


def test_head_tail_noop_when_under_limit():
    out, truncated = head_tail_truncate("short", 100)
    assert out == "short"
    assert truncated is False


def test_head_tail_stderr_truncation_keeps_stack_tail():
    """stderr 的错误堆栈在尾部——纯头部截断会丢掉它。"""
    text = "x" * 1000 + "\nTraceback: the real error"
    out, truncated = head_tail_truncate(text, 200)
    assert truncated is True
    assert "the real error" in out


# ── bash structured output ─────────────────────────────────────


async def test_bash_returns_structured_json(tmp_path):
    from connectclaw.coding.tools.bash import BashTool

    from connectclaw.coding.safety.sandbox import DirectSandbox
    from connectclaw.coding.tools.bash import BashGuard

    tool = BashTool.__new__(BashTool)  # 跳过 __init__ 的沙箱探测
    tool._cwd = str(tmp_path)
    tool._guard = BashGuard()
    tool._sandbox_cls = DirectSandbox
    tool._git_root = None
    r = await tool.execute("t", {"command": "echo hello-structured; echo err >&2; false"})
    payload = json.loads(r.content[0]["text"])
    assert "hello-structured" in payload["output"]
    assert "err" in payload["stderr"]
    assert payload["exit_code"] == 1
    assert "wall_time_s" in payload


# ── view_image detail hint ─────────────────────────────────────


async def test_view_image_detail_passthrough(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(_TINY_PNG)
    tool = create_view_image_tool(str(tmp_path))

    r = await tool.execute("t", {"path": "x.png", "detail": "low"})
    ref = next(b for b in r.content if b.get("type") == "image_ref")
    assert ref["detail"] == "low"

    # 非法值不进 block
    r = await tool.execute("t", {"path": "x.png", "detail": "ultra"})
    ref = next(b for b in r.content if b.get("type") == "image_ref")
    assert "detail" not in ref


def test_provider_forwards_detail_to_wire_format():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_TINY_PNG)
        path = f.name
    p = DeepSeekProvider()
    block = {"type": "image_ref", "path": path, "id": "x",
             "mime_type": "image/png", "size": len(_TINY_PNG), "detail": "low"}
    converted = DeepSeekProvider._convert_image_block(p, block, resolve=True)
    assert converted["type"] == "image_url"
    assert converted["image_url"]["detail"] == "low"
    assert converted["image_url"]["url"].startswith("data:image/png;base64,")

    # 无 detail 时不写该字段（网关 auto）
    block2 = dict(block)
    block2.pop("detail")
    converted2 = DeepSeekProvider._convert_image_block(p, block2, resolve=True)
    assert "detail" not in converted2["image_url"]
