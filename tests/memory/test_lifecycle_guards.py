"""Lifecycle guards for the memory system — regression tests for the
2026-09 diagnosis: extraction stalling silently, config-snapshot memories
self-promoting into the always-on persona block, and the double-decay bug.

All deterministic: no LLM, no embedding model, no network.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from connectclaw.memory import subsystem as subsystem_mod
from connectclaw.memory.store import MemoryStore
from connectclaw.memory.subsystem import MemoryConfig, MemorySubsystem
from connectclaw.memory.retriever import MemoryRetriever, RetrievalConfig
from connectclaw.memory.types import MemoryEntry, MemoryType, SearchResult


# ── extraction throttle ──────────────────────────────────────────


async def test_learn_fires_at_min_then_interval(tmp_path, monkeypatch):
    """learn() must fire at extract_min_turns and every
    extract_interval_turns after (3, 8, …) — the old `turns % interval` gate
    needed a whole extra interval before the first extraction."""
    calls: list[int] = []

    async def fake_extract(messages, model, **kwargs):
        calls.append(len(calls) + 1)
        return []

    monkeypatch.setattr(subsystem_mod, "extract_memories", fake_extract)

    sub = MemorySubsystem(
        MemoryConfig(use_embeddings=False, db_path=str(tmp_path / "m.db"))
    )
    await sub.initialize()
    for _ in range(10):
        await sub.learn([{"role": "user", "content": "hi"}], object(),
                        conversation_key="c")
    assert len(calls) == 2  # turns 3 and 8


async def test_learn_below_min_turns_never_fires(tmp_path):
    sub = MemorySubsystem(
        MemoryConfig(use_embeddings=False, db_path=str(tmp_path / "m.db"))
    )
    await sub.initialize()
    for _ in range(2):
        n = await sub.learn([{"role": "user", "content": "hi"}], object(),
                            conversation_key="c")
        assert n == 0


# ── persona promotion guard ──────────────────────────────────────


def _add(store, content, *, importance=0.5, strength=1.0):
    now = time.time()
    e = MemoryEntry(
        type=MemoryType.SEMANTIC, content=content, importance=importance,
        strength=strength, created_at=now, last_accessed=now,
    )
    store.add(e)
    return e


def test_confirm_usage_cannot_promote_into_persona(tmp_path):
    """Repeated confirmations raise importance toward the 0.65 ceiling —
    which stays below the persona floor (0.85). The old cap (0.8) let
    config-snapshot memories walk into the always-on block."""
    store = MemoryStore(str(tmp_path / "m.db"))
    e = _add(store, "当前激活的LLM配置是某快照描述")
    retriever = MemoryRetriever(store, RetrievalConfig())
    for _ in range(50):
        result = SearchResult(entry=store.get(e.id), score=0.6)
        retriever.confirm_usage("当前激活的LLM配置是某快照描述", [result])
    final = store.get(e.id).importance
    assert final == pytest.approx(0.65)
    assert final < RetrievalConfig().persona_min_importance


def test_persona_block_only_takes_true_identity(tmp_path):
    """0.95 identity facts stay always-on; 0.8 config snapshots stay out."""
    store = MemoryStore(str(tmp_path / "m.db"))
    hi = _add(store, "用户要求称呼其为主人", importance=0.95)
    lo = _add(store, "Vision模型配置为某快照", importance=0.8)
    retriever = MemoryRetriever(store, RetrievalConfig())
    ids = {r.entry.id for r in retriever._retrieve_persona()}
    assert hi.id in ids
    assert lo.id not in ids


# ── decay-anchor schema migration ────────────────────────────────


def test_last_decayed_at_migration_anchors_existing_rows(tmp_path):
    """An old-schema DB (pre last_decayed_at) opens cleanly; existing rows
    anchor at migration time so the next dream doesn't re-decay their full
    age."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE memories (
            id TEXT PRIMARY KEY, type TEXT NOT NULL, content TEXT NOT NULL,
            detail TEXT, category TEXT DEFAULT '', tags TEXT DEFAULT '[]',
            importance REAL DEFAULT 0.5, created_at REAL NOT NULL,
            last_accessed REAL NOT NULL, access_count INTEGER DEFAULT 0,
            embedding BLOB, source_session TEXT, strength REAL DEFAULT 1.0,
            metadata TEXT DEFAULT '{}')"""
    )
    now = time.time()
    conn.execute(
        "INSERT INTO memories (id, type, content, created_at, last_accessed)"
        " VALUES ('legacy1', 'semantic', 'old row', ?, ?)",
        (now, now),
    )
    conn.commit()
    conn.close()

    store = MemoryStore(str(db))
    entry = store.get("legacy1")
    assert entry is not None
    assert entry.last_decayed_at is not None
    assert entry.last_decayed_at > now - 60
    store.close()


def test_new_entries_roundtrip_last_decayed_at(tmp_path):
    store = MemoryStore(str(tmp_path / "m.db"))
    now = time.time()
    e = _add(store, "x", importance=0.5)
    entry = store.get(e.id)
    entry.last_decayed_at = now
    store.update(entry)
    assert store.get(e.id).last_decayed_at == pytest.approx(now)
    store.close()
