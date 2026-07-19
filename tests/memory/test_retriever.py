"""Tests for hybrid retrieval: type quota, BM25 fusion, and usage confirmation.

Covers the retrieval improvements:
- per-type floor so one memory class can't monopolize TopK
- BM25 signal fused with embedding similarity (exact-term matches surface)
- recall is not usage — only memories referenced in the reply get touched
"""

from __future__ import annotations

import pytest

from connectclaw.memory.retriever import MemoryRetriever, RetrievalConfig, _content_referenced
from connectclaw.memory.store import MemoryStore
from connectclaw.memory.types import MemoryEntry, MemoryType


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    yield s
    s.close()


@pytest.fixture
def retriever(store):
    return MemoryRetriever(store, RetrievalConfig(min_similarity=0.0, min_score=0.0))


def _make(store, content, embedding, mtype=MemoryType.SEMANTIC, importance=0.5):
    e = MemoryEntry(
        type=mtype, content=content, importance=importance,
        embedding=embedding,
    )
    store.add(e)
    return e


# ── type quota ───────────────────────────────────────────────

def test_type_quota_gives_each_type_a_floor(store, retriever):
    for i in range(5):
        _make(store, f"proc {i}", [1.0, 0.0, 0.0], mtype=MemoryType.PROCEDURAL)
    _make(store, "sem fact", [1.0, 0.0, 0.0], mtype=MemoryType.SEMANTIC)
    _make(store, "epi event", [1.0, 0.0, 0.0], mtype=MemoryType.EPISODIC)

    import asyncio
    results = asyncio.run(retriever.retrieve("x", query_embedding=[1.0, 0.0, 0.0]))
    types = [r.entry.type for r in results]
    assert MemoryType.SEMANTIC in types
    assert MemoryType.EPISODIC in types
    assert MemoryType.PROCEDURAL in types


def test_type_quota_does_not_overfill_when_sparse(store, retriever):
    for i in range(3):
        _make(store, f"proc {i}", [1.0, 0.0, 0.0], mtype=MemoryType.PROCEDURAL)
    import asyncio
    results = asyncio.run(retriever.retrieve("x", query_embedding=[1.0, 0.0, 0.0]))
    assert len(results) == 3


# ── BM25 fusion ──────────────────────────────────────────────

def test_bm25_surfaces_exact_term_match(store, retriever):
    _make(store, "用户的项目叫 ConnectClaw", [0.6, 0.1, 0.1])
    _make(store, "不相关的内容", [0.9, 0.0, 0.0])
    import asyncio
    results = asyncio.run(retriever.retrieve("ConnectClaw", query_embedding=[0.6, 0.1, 0.1]))
    contents = [r.entry.content for r in results]
    assert "用户的项目叫 ConnectClaw" in contents


# ── usage confirmation: recall != usage ──────────────────────

def test_retrieve_does_not_touch(store, retriever):
    e = _make(store, "用户喜欢深色主题", [1.0, 0.0, 0.0])
    before = store.get(e.id).access_count
    import asyncio
    asyncio.run(retriever.retrieve("深色主题", query_embedding=[1.0, 0.0, 0.0]))
    after = store.get(e.id).access_count
    assert after == before


def test_confirm_usage_touches_referenced(store, retriever):
    e = _make(store, "用户喜欢深色主题", [1.0, 0.0, 0.0])
    import asyncio
    results = asyncio.run(retriever.retrieve("深色主题", query_embedding=[1.0, 0.0, 0.0]))
    assert store.get(e.id).access_count == 0
    n = retriever.confirm_usage("好的，已切换到深色主题", results)
    assert n >= 1
    assert store.get(e.id).access_count == 1


def test_confirm_usage_skips_unreferenced(store, retriever):
    e = _make(store, "用户喜欢深色主题", [1.0, 0.0, 0.0])
    import asyncio
    results = asyncio.run(retriever.retrieve("深色主题", query_embedding=[1.0, 0.0, 0.0]))
    n = retriever.confirm_usage("今天天气不错", results)
    assert n == 0
    assert store.get(e.id).access_count == 0


def test_confirm_usage_persona_always_confirmed(store, retriever):
    e = _make(store, "请叫我老板", [1.0, 0.0, 0.0], importance=0.9)
    import asyncio
    text, results = asyncio.run(
        retriever.retrieve_formatted("hi", query_embedding=[1.0, 0.0, 0.0])
    )
    persona_results = [r for r in results if r.score >= 1.0]
    assert persona_results, "persona should be present"
    n = retriever.confirm_usage("好的明白了", results)
    assert n >= 1
    assert store.get(e.id).access_count >= 1


def test_confirm_usage_empty_inputs(retriever):
    assert retriever.confirm_usage("", []) == 0
    assert retriever.confirm_usage("some reply", []) == 0


# ── _content_referenced unit ─────────────────────────────────

def test_content_referenced_long_run():
    assert _content_referenced("用户喜欢深色主题", "好的，已切换到深色主题") is True


def test_content_referenced_no_match():
    assert _content_referenced("用户喜欢深色主题", "今天天气不错") is False


def test_content_referenced_short_content_whole_string():
    assert _content_referenced("ConnectClaw", "我用的 ConnectClaw 项目") is True


def test_content_referenced_single_common_word_not_enough():
    assert _content_referenced("用户的名字是张三", "我的天") is False
