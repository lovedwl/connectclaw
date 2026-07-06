"""SQLite-based memory store. Single file, no external dependencies."""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import time
import uuid
from typing import Any

import numpy as np

from connectclaw.logging import get_logger

from .types import MemoryEntry, MemoryType

logger = get_logger(__name__)


class MemoryStore:
    """Persistent memory storage backed by SQLite."""

    def __init__(self, db_path: str):
        self._db_path = os.path.expanduser(db_path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        conn = self._conn
        assert conn is not None
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                detail TEXT,
                category TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                embedding BLOB,
                source_session TEXT,
                strength REAL DEFAULT 1.0,
                metadata TEXT DEFAULT '{}'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_category ON memories(category)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_importance ON memories(importance DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_strength ON memories(strength)"
        )
        conn.commit()

    # ── CRUD ──────────────────────────────────────────────

    def add(self, entry: MemoryEntry) -> str:
        conn = self._connect()
        if not entry.id:
            entry.id = uuid.uuid4().hex[:12]
        if not entry.created_at:
            entry.created_at = time.time()
        if not entry.last_accessed:
            entry.last_accessed = entry.created_at

        conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, type, content, detail, category, tags, importance,
                created_at, last_accessed, access_count, embedding,
                source_session, strength, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.type.value if isinstance(entry.type, MemoryType) else entry.type,
                entry.content,
                entry.detail,
                entry.category,
                json.dumps(entry.tags, ensure_ascii=False),
                entry.importance,
                entry.created_at,
                entry.last_accessed,
                entry.access_count,
                _floats_to_blob(entry.embedding) if entry.embedding else None,
                entry.source_session,
                entry.strength,
                json.dumps(entry.metadata, ensure_ascii=False),
            ),
        )
        conn.commit()
        return entry.id

    def get(self, memory_id: str) -> MemoryEntry | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_entry(row)

    def update(self, entry: MemoryEntry) -> None:
        self.add(entry)

    def delete(self, memory_id: str) -> bool:
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def touch(self, memory_id: str) -> None:
        conn = self._connect()
        conn.execute(
            """UPDATE memories
               SET last_accessed = ?, access_count = access_count + 1
               WHERE id = ?""",
            (time.time(), memory_id),
        )
        conn.commit()

    # ── Queries ───────────────────────────────────────────

    def list_all(
        self,
        *,
        memory_type: MemoryType | None = None,
        category: str | None = None,
        min_strength: float = 0.0,
        limit: int = 1000,
    ) -> list[MemoryEntry]:
        conn = self._connect()
        query = "SELECT * FROM memories WHERE strength >= ?"
        params: list[Any] = [min_strength]

        if memory_type:
            query += " AND type = ?"
            params.append(
                memory_type.value
                if isinstance(memory_type, MemoryType)
                else memory_type
            )
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def search_by_embedding(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 20,
        min_strength: float = 0.1,
    ) -> list[tuple[MemoryEntry, float]]:
        """Find memories most similar to query embedding.

        Uses numpy cosine similarity — fast enough for thousands of entries.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM memories WHERE embedding IS NOT NULL AND strength >= ?",
            (min_strength,),
        ).fetchall()

        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        results: list[tuple[MemoryEntry, float]] = []
        for row in rows:
            entry = self._row_to_entry(row)
            if not entry.embedding:
                continue
            entry_vec = np.array(entry.embedding, dtype=np.float32)
            entry_norm = np.linalg.norm(entry_vec)
            if entry_norm == 0:
                continue
            similarity = float(
                np.dot(query_vec, entry_vec) / (query_norm * entry_norm)
            )
            results.append((entry, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_keywords(
        self, keywords: set[str], *, min_strength: float = 0.1, top_k: int = 20
    ) -> list[tuple[MemoryEntry, float]]:
        """Fallback keyword search when no embedding is available."""
        all_memories = self.list_all(min_strength=min_strength, limit=10000)
        if not all_memories:
            return []

        results: list[tuple[MemoryEntry, float]] = []
        for entry in all_memories:
            content_lower = entry.content.lower()
            hits = sum(1 for kw in keywords if kw.lower() in content_lower)
            if hits == 0:
                continue
            score = min(1.0, hits / max(len(keywords), 1)) * 0.5
            results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict[str, Any]:
        conn = self._connect()
        stats: dict[str, Any] = {}
        for mem_type in MemoryType:
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE type = ? AND strength >= 0.1",
                (mem_type.value,),
            ).fetchone()[0]
            stats[mem_type.value] = count
        stats["total"] = sum(stats.values())
        stats["db_size_kb"] = (
            os.path.getsize(self._db_path) // 1024
            if os.path.exists(self._db_path)
            else 0
        )
        return stats

    def cleanup(self, min_strength: float = 0.05) -> int:
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM memories WHERE strength < ?", (min_strength,)
        )
        conn.commit()
        return cursor.rowcount

    # ── Internal ──────────────────────────────────────────

    def _row_to_entry(self, row: tuple) -> MemoryEntry:
        return MemoryEntry(
            id=row[0],
            type=MemoryType(row[1]),
            content=row[2],
            detail=row[3],
            category=row[4],
            tags=json.loads(row[5]) if row[5] else [],
            importance=row[6],
            created_at=row[7],
            last_accessed=row[8],
            access_count=row[9],
            embedding=_blob_to_floats(row[10]) if row[10] else [],
            source_session=row[11],
            strength=row[12],
            metadata=json.loads(row[13]) if row[13] else {},
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ── Embedding serialization ──────────────────────────────────


def _floats_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_floats(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))
