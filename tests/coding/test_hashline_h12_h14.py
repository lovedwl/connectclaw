"""2026-10 (H12/H13/H14) regression tests — hashline silent-corruption fixes.

Covers:
- H12: CRLF files keep CRLF after hash_edit (newline="" on read/write)
- H13: replace + same-boundary insert no longer corrupt the replaced line
- H14: file mode (e.g. executable bit) survives atomic replace;
       three_way_merge preserves the trailing newline
"""

from __future__ import annotations

import os
import re

from connectclaw.coding.tools.hash_edit import create_hash_edit_tool
from connectclaw.coding.tools.hash_read import create_hash_read_tool
from connectclaw.hashline.merge import three_way_merge

ANCHOR_RE = re.compile(r"^\s*(\d+#[A-Z]{2}):", re.M)


def _anchors(text: str) -> dict[int, str]:
    return {int(m.group(1).split("#")[0]): m.group(1) for m in ANCHOR_RE.finditer(text)}


# ── H12: CRLF preservation ─────────────────────────────────────


async def test_hash_edit_preserves_crlf(tmp_path):
    f = tmp_path / "crlf.txt"
    f.write_bytes(b"aa\r\nbb\r\n")
    he = create_hash_edit_tool(str(tmp_path))
    r = await he.execute("t", {
        "path": str(f),
        "edits": [{"op": "replace_text", "oldText": "aa", "newText": "XX"}],
    })
    assert "错误" not in r.content[0]["text"], r.content[0]["text"]
    # 整文件行尾必须仍是 CRLF（此前被静默改写成 LF）
    assert f.read_bytes() == b"XX\r\nbb\r\n"


async def test_hash_read_preserves_crlf_content(tmp_path):
    f = tmp_path / "crlf.txt"
    f.write_bytes(b"aa\r\nbb\r\n")
    hr = create_hash_read_tool(str(tmp_path))
    r = await hr.execute("t", {"path": str(f)})
    text = r.content[0]["text"]
    assert "aa" in text and "bb" in text


# ── H13: replace + same-boundary insert ────────────────────────


async def test_replace_with_same_boundary_append(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("aa\nbb\ncc\n")
    cwd = str(tmp_path)
    hr = create_hash_read_tool(cwd)
    he = create_hash_edit_tool(cwd)

    r = await hr.execute("t", {"path": str(f)})
    anchors = _anchors(r.content[0]["text"])
    assert 1 in anchors

    r = await he.execute("t", {"path": str(f), "edits": [
        {"op": "replace", "pos": anchors[1], "lines": ["XXXX"]},
        {"op": "append", "pos": anchors[1], "lines": ["Q"]},
    ]})
    assert "错误" not in r.content[0]["text"], r.content[0]["text"]
    # 此前实际输出 "XX\nQXX\nbb\ncc\n"（替换行被截断错位）且返回成功
    assert f.read_text() == "XXXX\nQ\nbb\ncc\n"


async def test_same_boundary_append_then_replace_order(tmp_path):
    # 输入顺序互换结果必须一致
    f = tmp_path / "a.txt"
    f.write_text("aa\nbb\ncc\n")
    cwd = str(tmp_path)
    hr = create_hash_read_tool(cwd)
    he = create_hash_edit_tool(cwd)

    r = await hr.execute("t", {"path": str(f)})
    anchors = _anchors(r.content[0]["text"])

    r = await he.execute("t", {"path": str(f), "edits": [
        {"op": "append", "pos": anchors[1], "lines": ["Q"]},
        {"op": "replace", "pos": anchors[1], "lines": ["XXXX"]},
    ]})
    assert f.read_text() == "XXXX\nQ\nbb\ncc\n"


# ── H14: mode preservation + trailing newline ──────────────────


async def test_hash_edit_preserves_file_mode(tmp_path):
    f = tmp_path / "run.sh"
    f.write_text("#!/bin/sh\necho hi\n")
    os.chmod(f, 0o755)
    he = create_hash_edit_tool(str(tmp_path))
    r = await he.execute("t", {
        "path": str(f),
        "edits": [{"op": "replace_text", "oldText": "hi", "newText": "bye"}],
    })
    assert "错误" not in r.content[0]["text"], r.content[0]["text"]
    assert os.stat(f).st_mode & 0o777 == 0o755  # 可执行位不得被 umask 覆盖


async def test_write_preserves_file_mode(tmp_path):
    from connectclaw.coding.tools.read import create_read_tool
    from connectclaw.coding.tools.write import create_write_tool

    f = tmp_path / "run.sh"
    f.write_text("#!/bin/sh\necho hi\n")
    os.chmod(f, 0o755)
    rt = create_read_tool(str(tmp_path))
    await rt.execute("t", {"file_path": str(f)})  # read-before-write
    wt = create_write_tool(str(tmp_path), read_tool=rt)
    r = await wt.execute("t", {"file_path": str(f), "content": "#!/bin/sh\necho yo\n"})
    assert "错误" not in r.content[0]["text"], r.content[0]["text"]
    assert os.stat(f).st_mode & 0o777 == 0o755


def test_three_way_merge_preserves_trailing_newline():
    base = "aa\nbb\ncc\n"
    base_edited = "aa\nXX\ncc\n"
    current = "aa\nbb\ncc\nYY\n"
    merged = three_way_merge(base, base_edited, current)
    assert merged is not None
    assert merged == "aa\nXX\ncc\nYY\n"  # 此前丢失结尾 \n


def test_three_way_merge_no_trailing_newline_stays():
    # 无尾换行 + 补丁远在上下文窗口之外（n=3 需要上下文匹配）
    base = "\n".join(f"l{i}" for i in range(1, 11))
    base_edited = base.replace("l2", "XX", 1)
    current = "\n".join(f"l{i}" for i in range(1, 10)) + "\nYY"
    merged = three_way_merge(base, base_edited, current)
    assert merged == "\n".join(["l1", "XX", *[f"l{i}" for i in range(3, 10)], "YY"])
    assert not merged.endswith("\n")
