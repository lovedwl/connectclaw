"""Tests for the 2026-10 hashline tooling fixes (R1-R6).

See docs/hashline-tool-usage-report.md:
- R1: anchor block threshold 12 -> 48 (multi-line edits keep their anchors)
- R2: hash_read bridges to the legacy read tool's read-before-write ledger
- R3: tool error paths set details.is_error; agent loop honors it
- R4: E_RANGE_OOB includes the boundary line's current anchor
- R5: missing/truncated required params get explicit errors
- R6: hash_read flags read-only protected files up front
"""
from __future__ import annotations

import re

from connectclaw.agent.agent_loop import _result_is_error
from connectclaw.agent.types import AgentToolResult
from connectclaw.coding.tools.hash_edit import create_hash_edit_tool
from connectclaw.coding.tools.hash_read import create_hash_read_tool
from connectclaw.coding.tools.read import create_read_tool
from connectclaw.coding.tools.write import create_write_tool

ANCHOR_RE = re.compile(r"^\s*(\d+#[A-Z]{2}):", re.M)


def _anchors(text: str) -> dict[int, str]:
    return {int(m.group(1).split("#")[0]): m.group(1) for m in ANCHOR_RE.finditer(text)}


async def _hr(tool, path, **kw):
    r = await tool.execute("t", {"path": path, **kw})
    return r.content[0]["text"], r.details


# ── R1: anchor budget ──────────────────────────────────────────


async def test_multiline_edit_returns_anchor_block(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("\n".join(f"line-{i}" for i in range(1, 31)) + "\n")
    cwd = str(tmp_path)
    hr = create_hash_read_tool(cwd)
    he = create_hash_edit_tool(cwd)

    text, _ = await _hr(hr, str(f))
    a = _anchors(text)
    # replace 15 lines (5..19) — under the old 12-line budget this was omitted
    r = await he.execute(
        "t",
        {"path": str(f), "edits": [{
            "op": "replace", "pos": a[5], "end": a[19],
            "lines": [f"new-{i}" for i in range(1, 16)],
        }]},
    )
    out = r.content[0]["text"]
    assert "编辑已成功应用" in out
    assert "Anchors omitted" not in out
    assert "--- 锚点" in out
    # fresh anchors are usable for a follow-up edit without re-reading
    fresh = _anchors(out)
    assert fresh, "anchor block contained no usable anchors"


async def test_huge_edit_still_omits_anchors(tmp_path):
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line-{i}" for i in range(1, 71)) + "\n")
    cwd = str(tmp_path)
    hr = create_hash_read_tool(cwd)
    he = create_hash_edit_tool(cwd)

    text, _ = await _hr(hr, str(f))
    a = _anchors(text)
    r = await he.execute(
        "t",
        {"path": str(f), "edits": [{
            "op": "replace", "pos": a[3], "end": a[60],
            "lines": [f"new-{i}" for i in range(1, 59)],
        }]},
    )
    assert "Anchors omitted" in r.content[0]["text"]


# ── R3: is_error flagging ──────────────────────────────────────


async def test_hash_edit_error_paths_flagged(tmp_path):
    cwd = str(tmp_path)
    he = create_hash_edit_tool(cwd)

    # nonexistent file
    r = await he.execute("t", {"path": str(tmp_path / "nope.txt"),
                               "edits": [{"op": "append", "lines": ["x"]}]})
    assert r.details and r.details.get("is_error") is True

    # bad anchor
    f = tmp_path / "b.txt"
    f.write_text("one\ntwo\n")
    r = await he.execute("t", {"path": str(f),
                               "edits": [{"op": "replace", "pos": "99#zz", "lines": ["x"]}]})
    assert "E_BAD_REF" in r.content[0]["text"]
    assert r.details.get("is_error") is True

    # empty edits (R5: truncation hint)
    r = await he.execute("t", {"path": str(f), "edits": []})
    assert r.details.get("is_error") is True
    assert "截断" in r.content[0]["text"]

    # missing path (R5)
    r = await he.execute("t", {"edits": [{"op": "append", "lines": ["x"]}]})
    assert r.details.get("is_error") is True
    assert "file_path" not in r.content[0]["text"] and "path" in r.content[0]["text"]


async def test_hash_read_error_paths_flagged(tmp_path):
    cwd = str(tmp_path)
    hr = create_hash_read_tool(cwd)
    r = await hr.execute("t", {"path": str(tmp_path / "nope.txt")})
    assert r.details and r.details.get("is_error") is True

    f = tmp_path / "c.txt"
    f.write_text("one\ntwo\n")
    r = await hr.execute("t", {"path": str(f), "offset": 9999})
    assert r.details.get("is_error") is True


def test_result_is_error_helper():
    assert _result_is_error(AgentToolResult(content=[], details={"is_error": True})) is True
    assert _result_is_error(AgentToolResult(content=[], details=None)) is False
    assert _result_is_error(AgentToolResult(content=[], details={})) is False
    assert _result_is_error(AgentToolResult(content=[], details={"diff": "x"})) is False


# ── R4: OOB self-heal anchor ───────────────────────────────────


async def test_range_oob_includes_boundary_anchor(tmp_path):
    f = tmp_path / "d.txt"
    f.write_text("\n".join(f"line-{i}" for i in range(1, 11)) + "\n")
    he = create_hash_edit_tool(str(tmp_path))
    r = await he.execute("t", {"path": str(f),
                               "edits": [{"op": "replace", "pos": "99#ZM", "lines": ["x"]}]})
    out = r.content[0]["text"]
    assert "E_RANGE_OOB" in out
    assert "文件共有 10 行" in out
    # the last line's current anchor must be offered for one-step self-correction
    assert re.search(r"10#[A-Z]{2}:line-10", out)
    assert r.details.get("is_error") is True


# ── R2: hash_read -> write bridge ──────────────────────────────


async def test_hash_read_satisfies_write(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("old\n")
    cwd = str(tmp_path)
    rt = create_read_tool(cwd)
    ht = create_hash_read_tool(cwd, rt)
    wt = create_write_tool(cwd, rt)

    # control: never read -> refused
    r = await wt.execute("t", {"file_path": str(f), "content": "new\n"})
    assert "尚未读取" in r.content[0]["text"]
    assert r.details.get("is_error") is True

    # hash_read bridges the ledger -> write succeeds
    await ht.execute("t", {"path": str(f)})
    r = await wt.execute("t", {"file_path": str(f), "content": "new\n"})
    assert "成功写入" in r.content[0]["text"]
    assert f.read_text() == "new\n"


# ── R6: protected file note ────────────────────────────────────


async def test_hash_read_protected_note(tmp_path, monkeypatch):
    import connectclaw.coding.tools.hash_read as hr_mod
    f = tmp_path / "f.txt"
    f.write_text("secret\n")
    monkeypatch.setattr(hr_mod, "is_protected", lambda p: True)
    hr = create_hash_read_tool(str(tmp_path))
    text, _ = await _hr(hr, str(f))
    assert "只读受保护" in text
    assert "secret" in text  # content still readable


async def test_hash_edit_protected_flagged(tmp_path, monkeypatch):
    import connectclaw.coding.tools.hash_edit as he_mod
    f = tmp_path / "g.txt"
    f.write_text("secret\n")
    monkeypatch.setattr(he_mod, "is_protected", lambda p: True)
    he = create_hash_edit_tool(str(tmp_path))
    r = await he.execute("t", {"path": str(f),
                               "edits": [{"op": "append", "lines": ["x"]}]})
    assert "受保护" in r.content[0]["text"]
    assert r.details.get("is_error") is True
    assert f.read_text() == "secret\n"
