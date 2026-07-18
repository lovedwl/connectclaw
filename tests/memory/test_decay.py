"""Tests for memory decay & strengthening — the "forgetting curve" path.

Covers decay behavior, the "used memories should not be forgotten"
guarantee, and the strengthening loop. All deterministic, no LLM, no
embeddings model.
"""

from __future__ import annotations

import time

import pytest

from connectclaw.memory.consolidator import ConsolidationConfig, MemoryConsolidator
from connectclaw.memory.store import MemoryStore
from connectclaw.memory.types import MemoryEntry, MemoryType


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    yield s
    s.close()


@pytest.fixture
def consolidator(store):
    return MemoryConsolidator(store, ConsolidationConfig(decay_halflife_days=30.0))


def _make(store, *, content="x", strength=1.0, importance=0.5,
          last_accessed=None, access_count=0, mtype=MemoryType.SEMANTIC):
    e = MemoryEntry(
        type=mtype,
        content=content,
        importance=importance,
        strength=strength,
        last_accessed=last_accessed if last_accessed is not None else time.time(),
        access_count=access_count,
    )
    store.add(e)
    return e


def test_fresh_memory_unchanged(consolidator, store):
    """A memory accessed moments ago should not be decayed."""
    e = _make(store, strength=1.0)
    n = consolidator.apply_decay_only()
    assert n == 0
    assert store.get(e.id).strength == pytest.approx(1.0, abs=0.01)


def test_decay_after_one_halflife(consolidator, store):
    """After one halflife (30d) strength should ~halve, floored at importance*0.3."""
    e = _make(store, strength=1.0, importance=0.2,
              last_accessed=time.time() - 30 * 86400)
    consolidator.apply_decay_only()
    new = store.get(e.id).strength
    # halflife decay: 1.0 * 0.5 = 0.5, well above floor 0.2*0.3=0.06
    assert new == pytest.approx(0.5, abs=0.05)


def test_decay_floor(consolidator, store):
    """Decay must not go below importance * 0.3 — never fully forget a valued memory."""
    e = _make(store, strength=0.2, importance=0.8,
              last_accessed=time.time() - 365 * 86400)
    consolidator.apply_decay_only()
    floor = 0.8 * 0.3
    assert store.get(e.id).strength >= floor - 0.001


def test_decay_is_by_last_accessed_not_created(consolidator, store):
    """A memory created long ago but accessed recently should stay strong."""
    now = time.time()
    e = _make(store, strength=1.0, importance=0.5,
              last_accessed=now - 1 * 86400)  # accessed yesterday
    e.created_at = now - 365 * 86400          # created a year ago
    store.update(e)
    consolidator.apply_decay_only()
    assert store.get(e.id).strength > 0.95


def test_used_memory_is_strengthened(consolidator, store):
    """The 'used → not forgotten' guarantee: access_count > 0 boosts strength.

    A memory that has been used must not follow the pure forgetting curve.
    Here a slightly decayed memory gets boosted back up.
    """
    e = _make(store, strength=0.6, importance=0.5, access_count=5)
    consolidator._boost_accessed()
    assert store.get(e.id).strength > 0.6


def test_unused_memory_not_boosted(consolidator, store):
    _make(store, strength=0.6, access_count=0)
    n = consolidator._boost_accessed()
    assert n == 0


def test_boost_caps_at_one(consolidator, store):
    """Strength can never exceed 1.0 even with huge access_count."""
    _make(store, strength=0.9, access_count=1000)
    consolidator._boost_accessed()
    # need the entry id
    entries = store.list_all()
    assert all(e.strength <= 1.0 + 1e-6 for e in entries)


def test_used_then_decayed_still_above_pure_decay(consolidator, store):
    """End-to-end: a used memory, after decay+boost, ends stronger than a
    never-used twin of the same age — the core property of the
    used-memory strengthening path.
    """
    age = 60 * 86400  # 60 days ago
    used = _make(store, content="used", strength=1.0, importance=0.5,
                 last_accessed=time.time() - age, access_count=8)
    cold = _make(store, content="cold", strength=1.0, importance=0.5,
                 last_accessed=time.time() - age, access_count=0)

    consolidator.apply_decay_only()
    consolidator._boost_accessed()

    assert store.get(used.id).strength > store.get(cold.id).strength


def test_cleanup_removes_below_threshold(consolidator, store):
    """Memories below decay_min_strength are deleted during cleanup."""
    _make(store, strength=0.01, importance=0.01,
          last_accessed=time.time() - 365 * 86400)
    consolidator.apply_decay_only()
    removed = store.cleanup(0.05)
    assert removed >= 1
    assert store.list_all() == []


def test_dream_runs_full_cycle_without_llm(consolidator, store):
    """dream() with no old episodic memories should still decay+boost+cleanup
    and return a report — no LLM path triggered."""
    _make(store, strength=1.0, importance=0.5,
          last_accessed=time.time() - 60 * 86400, access_count=3,
          mtype=MemoryType.SEMANTIC)
    import asyncio
    report = asyncio.run(consolidator.dream(model=None))
    assert report.decayed >= 1
    assert report.strengthened >= 1
