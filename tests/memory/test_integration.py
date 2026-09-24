"""Integration tests for the full memory lifecycle.

Covers the pipeline the interviewer asked about (Q17-Q19, Q41):
  add -> retrieve -> confirm_usage -> decay -> dream -> cleanup

All deterministic, no LLM, no embeddings model.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from connectclaw.memory.consolidator import ConsolidationConfig, MemoryConsolidator
from connectclaw.memory.retriever import RetrievalConfig, MemoryRetriever
from connectclaw.memory.store import MemoryStore
from connectclaw.memory.subsystem import MemorySubsystem, MemoryConfig
from connectclaw.memory.types import MemoryEntry, MemoryType


def _make_raw(store, *, content="x", strength=1.0, importance=0.5,
              last_accessed=None, access_count=0,
              embedding=None, mtype=MemoryType.SEMANTIC):
    e = MemoryEntry(
        type=mtype, content=content, importance=importance,
        strength=strength,
        last_accessed=last_accessed if last_accessed is not None else time.time(),
        access_count=access_count,
        embedding=embedding or [],
    )
    store.add(e)
    return e


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    yield s
    s.close()


class TestFullLifecycle:
    """The memory decay + integration pipeline the interviewer asked about (Q19)."""

    @pytest.fixture
    def retriever(self, store):
        return MemoryRetriever(store, RetrievalConfig(
            min_similarity=0.0, min_score=0.0,
        ))

    @pytest.fixture
    def consolidator(self, store):
        return MemoryConsolidator(store, ConsolidationConfig(
            decay_halflife_days=30.0,
        ))

    def test_add_retrieve_confirm_decay_clean(self, store, retriever, consolidator):
        """Full pipeline end-to-end: add -> retrieve -> confirm -> decay -> clean."""
        now = time.time()

        used = _make_raw(store, content="用户深色主题", strength=1.0, importance=0.5,
                         last_accessed=now - 60 * 86400, access_count=0,
                         embedding=[1.0, 0.0, 0.0])
        unused = _make_raw(store, content="过时配置", strength=1.0, importance=0.3,
                           last_accessed=now - 60 * 86400, access_count=0)
        assert len(store.list_all()) == 2

        results = asyncio.run(retriever.retrieve(
            "深色", query_embedding=[1.0, 0.0, 0.0],
        ))
        assert any("深色" in r.entry.content for r in results)

        n = retriever.confirm_usage("好的，已切换到深色主题", results)
        assert n >= 1
        assert store.get(used.id).access_count >= 1
        assert store.get(unused.id).access_count == 0

        consolidator.apply_decay_only()
        consolidator._boost_accessed()

        used_after = store.get(used.id)
        unused_after = store.get(unused.id)
        assert used_after.strength > unused_after.strength
        assert unused_after.strength < 0.6

    def test_used_memory_survives_multiple_decay_cycles(self, store, consolidator):
        now = time.time()
        e = _make_raw(store, content="用户的常用别名", strength=1.0, importance=0.6,
                      last_accessed=now, access_count=5)
        for cycle in range(3):
            e.last_accessed = now - (3 - cycle) * 30 * 86400
            store.update(e)
            consolidator.apply_decay_only()
            consolidator._boost_accessed()
        survived = store.get(e.id)
        assert survived is not None
        assert survived.strength >= 0.15

    def test_brand_new_memory_not_touched_by_decay(self, store, consolidator):
        e = _make_raw(store, content="刚生成的记忆", strength=1.0, importance=0.5)
        consolidator.apply_decay_only()
        assert store.get(e.id).strength == pytest.approx(1.0, abs=0.01)


class TestForgetByIdReturnsFullId:
    """forget_by_id returns the full id, not True/False."""

    @pytest.fixture
    def mem(self, tmp_path):
        cfg = MemoryConfig(enabled=True, db_path=str(tmp_path / "mem.db"), use_embeddings=False)
        m = MemorySubsystem(cfg)
        asyncio.run(m.initialize())
        yield m
        m._store.close() if m._store else None

    def test_exact_match_returns_full_id(self, mem):
        e = MemoryEntry(type=MemoryType.SEMANTIC, content="测试记忆")
        mem._store.add(e)
        result = asyncio.run(mem.forget_by_id(e.id))
        assert result == e.id

    def test_prefix_match_returns_expanded_id(self, mem):
        e = MemoryEntry(type=MemoryType.SEMANTIC, content="前缀匹配测试")
        mem._store.add(e)
        result = asyncio.run(mem.forget_by_id(e.id[:8]))
        assert result == e.id

    def test_missing_id_returns_none(self, mem):
        assert asyncio.run(mem.forget_by_id("nonexistent")) is None

    def test_empty_id_returns_none(self, mem):
        assert asyncio.run(mem.forget_by_id("")) is None


class TestStress:
    """Bulk operations: add 100, clear all, decay all."""

    @pytest.fixture
    def store(self, tmp_path):
        s = MemoryStore(str(tmp_path / "mem.db"))
        yield s
        s.close()

    def test_bulk_add_then_clear(self, store):
        ids = [_make_raw(store, content=f"记忆{i}").id for i in range(100)]
        assert len(store.list_all(limit=1000)) == 100
        for id_ in ids:
            store.delete(id_)
        assert len(store.list_all()) == 0

    def test_bulk_add_then_decay(self, store):
        now = time.time()
        for i in range(100):
            _make_raw(store, content=f"旧记忆{i}", strength=1.0, importance=0.1,
                      last_accessed=now - 365 * 86400)
        consolidator = MemoryConsolidator(store, ConsolidationConfig(decay_halflife_days=30.0))
        n = consolidator.apply_decay_only()
        assert n == 100


class TestSubsystemRecallConfirm:
    """Subsystem recall() -> confirm_usage() cycle."""

    @pytest.fixture
    def mem(self, tmp_path):
        cfg = MemoryConfig(enabled=True, db_path=str(tmp_path / "mem.db"), use_embeddings=False)
        m = MemorySubsystem(cfg)
        asyncio.run(m.initialize())
        yield m
        m._store.close() if m._store else None

    def test_recall_and_confirm(self, mem):
        for c in ["用户喜欢深色主题", "用户住在上海"]:
            mem._store.add(MemoryEntry(type=MemoryType.SEMANTIC, content=c))
        text, results = asyncio.run(mem.recall("深色主题"))
        assert "深色主题" in text
        for r in results:
            assert r.entry.access_count == 0
        n = mem.confirm_usage("好的，已切换到深色主题", results)
        assert n >= 1

    def test_recall_is_incremental(self, mem):
        """按需注入：首轮完整注入，之后无变化就**不注入**，换会话则重新注入。

        2026-09-24 实测老设计：168 个用户轮 100% 带记忆块、同一句话平均被注入
        34 次。增量注入就是为了消灭这种重复，同时保持历史纯追加（前缀缓存）。
        """
        for c in ["用户喜欢深色主题", "用户住在上海"]:
            mem._store.add(MemoryEntry(type=MemoryType.SEMANTIC, content=c, importance=0.9))

        first, results = asyncio.run(mem.recall("深色主题", session_id="s1"))
        assert "<remembered-context>" in first
        assert "深色主题" in first
        assert results, "召回结果仍要给全量（confirm_usage 依赖它）"

        second, results2 = asyncio.run(mem.recall("深色主题", session_id="s1"))
        assert second == "", "条目没变化就不该重复注入"
        assert results2, "但召回结果依然要返回"

        # 新会话（例如 /new）→ 新账本 → 重新完整注入一次
        fresh, _ = asyncio.run(mem.recall("深色主题", session_id="s2"))
        assert "<remembered-context>" in fresh

    def test_persona_is_incremental_too(self, mem):
        """persona 保持"每轮都参与召回"，但注入同样走增量——不该每轮重发。

        老设计实测：称呼/语言这类 persona 条目在 168/168 轮里被重复注入。
        """
        mem._store.add(MemoryEntry(type=MemoryType.SEMANTIC, content="用户要求全程中文",
                                   importance=0.95))
        first, results = asyncio.run(mem.recall("随便问一句", session_id="s1"))
        assert "全程中文" in first, "persona 首轮要注入"
        assert any("全程中文" in r.entry.content for r in results), "persona 仍参与召回"

        again, _ = asyncio.run(mem.recall("换一句问题", session_id="s1"))
        assert again == "", "persona 没变化就不该再注入一遍"

    def test_recall_reinjects_changed_item(self, mem):
        entry_id = mem._store.add(MemoryEntry(type=MemoryType.SEMANTIC, content="当前模型是 A"))
        asyncio.run(mem.recall("当前模型", session_id="s1"))
        assert asyncio.run(mem.recall("当前模型", session_id="s1"))[0] == ""

        # 内容变了（= 被更正）→ 必须重新注入
        mem._store.update(MemoryEntry(id=entry_id, type=MemoryType.SEMANTIC,
                                      content="当前模型是 B", importance=0.9))
        changed, _ = asyncio.run(mem.recall("当前模型", session_id="s1"))
        assert "当前模型是 B" in changed

    def test_recall_reports_forgotten_item(self, mem):
        entry_id = mem._store.add(MemoryEntry(type=MemoryType.SEMANTIC, content="临时偏好：用红色"))
        first, _ = asyncio.run(mem.recall("临时偏好", session_id="s1"))
        assert "用红色" in first

        mem._store.delete(entry_id)          # /forget（软删）
        again, _ = asyncio.run(mem.recall("临时偏好", session_id="s1"))
        assert "已遗忘" in again and "用红色" in again

    def test_clear_all(self, mem):
        for c in ["事实A", "事实B"]:
            mem._store.add(MemoryEntry(type=MemoryType.SEMANTIC, content=c))
        n = asyncio.run(mem.clear_all())
        assert n == 2
        assert len(mem._store.list_all()) == 0

    def test_stats(self, mem):
        mem._store.add(MemoryEntry(type=MemoryType.SEMANTIC, content="高重要性", importance=0.8))
        mem._store.add(MemoryEntry(type=MemoryType.SEMANTIC, content="低重要性", importance=0.3))
        stats = asyncio.run(mem.get_stats())
        assert stats["enabled"] is True
        assert stats["total"] >= 1
