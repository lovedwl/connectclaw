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


# ── 注入格式：新鲜度前缀 ─────────────────────────────────────

def test_format_for_prompt_includes_freshness_stamp(store, retriever):
    """每条记忆要带 [日期 · 强度]。

    库里本来就有 created_at / strength，但过去一个字都不输出——同主题的旧说法
    与新说法于是以同等权威并列（实测「RAG 已启用」被注入 84 次、当天写的更正
    只 1 次），模型没有任何依据判新旧，只能凭语序猜。
    """
    import asyncio
    import time as _time

    created = _time.mktime(_time.strptime("2026-07-27", "%Y-%m-%d"))
    store.add(MemoryEntry(
        type=MemoryType.SEMANTIC, content="当前激活LLM配置：dots3-note-prev",
        importance=0.9, embedding=[1.0, 0.0, 0.0], created_at=created, strength=0.31,
    ))

    results = asyncio.run(retriever.retrieve("激活", query_embedding=[1.0, 0.0, 0.0]))
    text = retriever._format_for_prompt(results)

    expected_day = _time.strftime("%Y-%m-%d", _time.localtime(created))
    assert f"[{expected_day} · 0.31]" in text
    assert "强度" in text  # 表头解释了这两个数字的含义
    assert text.startswith("<remembered-context>")
    assert text.rstrip().endswith("</remembered-context>")


def test_freshness_stamp_needs_created_at():
    from connectclaw.memory.retriever import _freshness_stamp

    assert _freshness_stamp(MemoryEntry(content="x", created_at=0.0)) == ""
    stamp = _freshness_stamp(MemoryEntry(content="x", created_at=1_700_000_000, strength=None))
    assert stamp.startswith("[") and stamp.endswith("] ")
    assert "1.00" in stamp  # strength 缺失时按 1.0，不编造别的数


# ── 召回"多余"的两道闸（2026-09-25 用户反馈后加）──────────────
#
# 实测：本机 BGE 中文余弦挤在 0.5~0.85 窄带里——无关项也有 0.52，而明显同一件事的
# 两种说法才 0.855。绝对阈值卡不出边界（min_similarity=0.48 形同虚设，一轮放 15 条）。
# 于是改成：① 跟本轮 top 比（低于 top×keep_ratio 丢掉，BM25 精确命中豁免）；
# ② 条数硬上限。

def test_relative_floor_drops_weak_hits(store, retriever):
    import asyncio
    _make(store, "强命中", [1.0, 0.0, 0.0])              # sim 1.00 → 保
    _make(store, "次强命中", [0.9, 0.4359, 0.0])          # sim ≈0.90 → 保
    _make(store, "边上但不相关", [0.7, 0.7141, 0.0])      # sim ≈0.70 → 丢（<0.85）

    results = asyncio.run(retriever.retrieve("x", query_embedding=[1.0, 0.0, 0.0]))
    contents = [r.entry.content for r in results]
    assert "强命中" in contents and "次强命中" in contents
    assert "边上但不相关" not in contents


def test_bm25_exact_hit_survives_relative_floor(store, retriever):
    """精确词命中的低相似度条目要留下——BM25 是特意融合的另一路信号。"""
    import asyncio
    _make(store, "ConnectClaw 的架构说明", [0.2, 0.9, 0.0])   # 低相似但含精确词
    _make(store, "完全无关的高相似内容", [1.0, 0.0, 0.0])

    results = asyncio.run(retriever.retrieve("ConnectClaw", query_embedding=[1.0, 0.0, 0.0]))
    contents = [r.entry.content for r in results]
    assert "ConnectClaw 的架构说明" in contents


def test_recall_cap_limits_items_per_turn(store, retriever):
    import asyncio
    for i in range(12):
        _make(store, f"记忆{i}", [1.0, 0.0, 0.0])
    results = asyncio.run(retriever.retrieve("x", query_embedding=[1.0, 0.0, 0.0]))
    assert len(results) <= retriever._config.recall_top_k
    assert retriever._config.recall_top_k < 12


def test_cap_is_configurable(store):
    from connectclaw.memory.retriever import MemoryRetriever, RetrievalConfig
    import asyncio
    r = MemoryRetriever(store, RetrievalConfig(min_similarity=0.0, min_score=0.0, recall_top_k=2))
    for i in range(6):
        _make(store, f"条目{i}", [1.0, 0.0, 0.0])
    assert len(asyncio.run(r.retrieve("x", query_embedding=[1.0, 0.0, 0.0]))) <= 2
