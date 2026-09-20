"""Tests for selective forget + agent-facing memory tool.

Covers:
- /forget by keyword / by id / by type, each with persona protection
- the memory tool's search (read-only) and forget (soft-retire) actions
- persona-grade memories are protected from bulk/agent removal but fall to
  an explicit forget-by-id
"""

from __future__ import annotations

import asyncio

import pytest

from connectclaw.coding.tools.memory import MemoryTool
from connectclaw.memory.subsystem import MemorySubsystem, MemoryConfig


def _memcfg(db_path: str) -> MemoryConfig:
    return MemoryConfig(
        enabled=True,
        db_path=db_path,
        use_embeddings=False,  # no model load in tests
    )


@pytest.fixture
def mem(tmp_path):
    m = MemorySubsystem(_memcfg(str(tmp_path / "mem.db")))
    asyncio.run(m.initialize())
    yield m
    m._store.close() if m._store else None


def _add(mem, content, *, mtype="semantic", importance=0.5, strength=1.0):
    from connectclaw.memory.types import MemoryEntry, MemoryType
    e = MemoryEntry(
        type=MemoryType(mtype), content=content,
        importance=importance, strength=strength,
    )
    mem._store.add(e)
    return e


# ── forget by keyword ────────────────────────────────────────

def test_forget_by_keyword_deletes_matches(mem):
    _add(mem, "用户喜欢深色主题")
    _add(mem, "深色主题护眼")
    _add(mem, "用户住在上海")
    n = asyncio.run(mem.forget_by_keyword("深色主题"))
    assert n == 2
    remaining = [e.content for e in mem._store.list_all()]
    assert "用户住在上海" in remaining
    assert not any("深色" in c for c in remaining)


def test_forget_by_keyword_protects_persona(mem):
    # importance >= 0.85 semantic = persona-grade, protected (0.8 config
    # snapshots are deliberately NOT persona — they must stay forgettable)
    _add(mem, "请叫我老板", importance=0.9)
    _add(mem, "老板喜欢深色", importance=0.4)
    n = asyncio.run(mem.forget_by_keyword("老板"))
    assert n == 1  # only the non-persona one
    contents = [e.content for e in mem._store.list_all(min_strength=0.0)]
    assert "请叫我老板" in contents


def test_forget_by_keyword_empty(mem):
    _add(mem, "x")
    assert asyncio.run(mem.forget_by_keyword("")) == 0


# ── forget by id ─────────────────────────────────────────────

def test_forget_by_id_deletes(mem):
    e = _add(mem, "一些过时的事实")
    ok = asyncio.run(mem.forget_by_id(e.id))
    assert ok == e.id
    assert mem._store.get(e.id) is None


def test_forget_by_id_bypasses_persona_protection(mem):
    # explicit id is an explicit decision — persona protection does NOT apply
    e = _add(mem, "请叫我老板", importance=0.9)
    ok = asyncio.run(mem.forget_by_id(e.id))
    assert ok == e.id
    assert mem._store.get(e.id) is None


def test_forget_by_id_missing(mem):
    assert asyncio.run(mem.forget_by_id("nonexistent")) is None


# ── forget by type ───────────────────────────────────────────

def test_forget_by_type(mem):
    _add(mem, "事实A", mtype="semantic")
    _add(mem, "事件B", mtype="episodic")
    _add(mem, "事件C", mtype="episodic")
    n = asyncio.run(mem.forget_by_type("episodic"))
    assert n == 2
    remaining = [e.type.value for e in mem._store.list_all()]
    assert remaining == ["semantic"]


def test_forget_by_type_invalid(mem):
    _add(mem, "x")
    assert asyncio.run(mem.forget_by_type("nonsense")) == 0


def test_forget_by_type_protects_persona(mem):
    _add(mem, "身份记忆", mtype="semantic", importance=0.9)
    _add(mem, "普通事实", mtype="semantic", importance=0.4)
    n = asyncio.run(mem.forget_by_type("semantic"))
    assert n == 1  # persona protected
    contents = [e.content for e in mem._store.list_all(min_strength=0.0)]
    assert "身份记忆" in contents


# ── soft forget (agent tool path) ────────────────────────────

def test_soften_by_keyword_zeros_strength(mem):
    e = _add(mem, "过时的项目名", strength=1.0)
    n = asyncio.run(mem.soften_by_keyword("项目名"))
    assert n == 1
    assert mem._store.get(e.id).strength == 0.0
    # still in store (reclaimable until cleanup)
    assert mem._store.get(e.id) is not None


def test_soften_protects_persona(mem):
    _add(mem, "请叫我老板", importance=0.9, strength=1.0)
    _add(mem, "老板的项目", importance=0.3, strength=1.0)
    n = asyncio.run(mem.soften_by_keyword("老板"))
    assert n == 1
    for e in mem._store.list_all(min_strength=0.0):
        if e.content == "请叫我老板":
            assert e.strength == 1.0  # untouched


def test_soften_idempotent(mem):
    e = _add(mem, "过时信息", strength=1.0)
    asyncio.run(mem.soften_by_keyword("过时"))
    n = asyncio.run(mem.soften_by_keyword("过时"))  # already 0
    assert n == 0


# ── MemoryTool (agent-facing) ────────────────────────────────

def test_tool_search_returns_matches(mem):
    _add(mem, "用户喜欢深色主题")
    _add(mem, "用户住在上海")
    tool = MemoryTool(mem)
    res = asyncio.run(tool.execute("t1", {"action": "search", "keyword": "深色"}))
    text = res.content[0]["text"]
    assert "深色主题" in text
    assert "住在上海" not in text


def test_tool_search_no_match(mem):
    tool = MemoryTool(mem)
    res = asyncio.run(tool.execute("t1", {"action": "search", "keyword": "不存在"}))
    assert "No memories" in res.content[0]["text"]


def test_tool_forget_soft_retires(mem):
    e = _add(mem, "过时的项目名", strength=1.0)
    tool = MemoryTool(mem)
    res = asyncio.run(tool.execute("t1", {"action": "forget", "keyword": "项目名"}))
    text = res.content[0]["text"]
    assert "Soft-retired 1" in text
    assert mem._store.get(e.id).strength == 0.0


def test_tool_forget_reports_persona_protection(mem):
    _add(mem, "请叫我老板", importance=0.9, strength=1.0)
    tool = MemoryTool(mem)
    res = asyncio.run(tool.execute("t1", {"action": "forget", "keyword": "老板"}))
    text = res.content[0]["text"]
    assert "protected" in text
    assert "/forget id" in text


def test_tool_unknown_action(mem):
    tool = MemoryTool(mem)
    res = asyncio.run(tool.execute("t1", {"action": "nuke", "keyword": "x"}))
    assert "Unknown action" in res.content[0]["text"]


def test_tool_missing_keyword(mem):
    tool = MemoryTool(mem)
    res = asyncio.run(tool.execute("t1", {"action": "search", "keyword": ""}))
    assert "keyword is required" in res.content[0]["text"]


def test_tool_disabled_subsystem():
    cfg = MemoryConfig(enabled=False)
    m = MemorySubsystem(cfg)
    tool = MemoryTool(m)
    res = asyncio.run(tool.execute("t1", {"action": "search", "keyword": "x"}))
    assert "disabled" in res.content[0]["text"]
