"""Memory subsystem — single entry point for the memory system.

Orchestrates extraction, retrieval, and consolidation.
Integrates with CodingAgent and AgentHarness.

Cache-friendly design: memory context is returned as a string to be
prepended to the user message, NOT injected into system prompt.
This preserves prompt cache hits (system prompt stays stable).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from connectclaw.logging import get_logger
from connectclaw.provider.types import Model

from .consolidator import ConsolidationConfig, MemoryConsolidator
from .extractor import extract_memories
from .retriever import MemoryRetriever, RetrievalConfig
from .store import MemoryStore
from .types import MemoryEntry, MemoryType

logger = get_logger(__name__)


@dataclass
class MemoryConfig:
    """Top-level memory configuration."""
    enabled: bool = True
    db_path: str = "~/.connectclaw/memory.db"
    extract_after_turn: bool = True
    extract_min_turns: int = 3
    extract_interval_turns: int = 5
    max_context_tokens: int = 2000
    recency_threshold_days: int = 7
    use_embeddings: bool = True
    dream_interval_hours: float = 24.0
    decay_halflife_days: float = 30.0
    consolidation_enabled: bool = True


class MemorySubsystem:
    """Unified memory subsystem. Lazy-initialized, all no-ops if disabled."""

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._store: MemoryStore | None = None
        self._retriever: MemoryRetriever | None = None
        self._consolidator: MemoryConsolidator | None = None
        self._embedding_provider = None
        self._initialized = False
        self._turn_counter: dict[str, int] = {}
        self._last_dream_time: float = 0.0
        self._dream_task: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    async def initialize(self) -> None:
        """Initialize the memory subsystem."""
        if self._initialized or not self.enabled:
            self._initialized = True
            return

        self._store = MemoryStore(self._config.db_path)

        self._retriever = MemoryRetriever(
            self._store,
            RetrievalConfig(
                max_context_tokens=self._config.max_context_tokens,
                recency_threshold_days=self._config.recency_threshold_days,
            ),
        )

        self._consolidator = MemoryConsolidator(
            self._store,
            ConsolidationConfig(
                decay_halflife_days=self._config.decay_halflife_days,
            ),
        )

        if self._config.use_embeddings:
            try:
                from connectclaw.provider.embedding import (
                    get_shared_embedding_provider,
                )

                # Share the single BGE-M3 instance with RAG — avoids loading
                # the ~2GB model into memory twice.
                self._embedding_provider = get_shared_embedding_provider()
                logger.info("Memory: using shared embedding provider")
            except ImportError:
                logger.info(
                    "Memory: no embedding provider, using keyword retrieval"
                )
                self._embedding_provider = None

        logger.info("Memory subsystem initialized: %s", self._config.db_path)
        self._initialized = True

    # ── Retrieval (called before each turn) ───────────────

    async def recall(
        self,
        query: str,
        *,
        conversation_key: str = "",
    ) -> str:
        """Retrieve relevant memories for the current conversation.

        Returns formatted text to prepend to user message (cache-friendly),
        or "" if nothing relevant.
        """
        if not self._initialized:
            await self.initialize()
        if not self._retriever:
            return ""

        query_embedding = None
        if self._embedding_provider:
            try:
                query_embedding = await self._embedding_provider.embed_query(query)
            except Exception as e:
                logger.debug("Memory: embedding failed for query: %s", e)

        # Memory is best-effort: a retrieval error must never break the turn.
        try:
            return await self._retriever.retrieve_formatted(
                query, query_embedding=query_embedding
            )
        except Exception as e:
            logger.debug("Memory: recall failed: %s", e)
            return ""

    # ── Extraction (called after each turn) ───────────────

    async def learn(
        self,
        messages: list[dict[str, Any]],
        model: Model,
        *,
        api_key: str | None = None,
        conversation_key: str = "",
        session_id: str | None = None,
    ) -> int:
        """Extract and store memories from a conversation turn.

        Returns the number of new memories stored.
        Throttled: only runs every N turns to save API costs.
        """
        if not self._initialized:
            await self.initialize()
        if not self._store:
            return 0

        turns = self._turn_counter.get(conversation_key, 0) + 1
        self._turn_counter[conversation_key] = turns

        if turns < self._config.extract_min_turns:
            return 0
        if turns % self._config.extract_interval_turns != 0:
            return 0

        # Extraction runs in a background task; isolate all failures so a bad
        # LLM response or DB error never surfaces to the user.
        try:
            existing = self._store.list_all(limit=100)

            new_entries = await extract_memories(
                messages,
                model,
                api_key=api_key,
                existing_memories=existing,
                source_session=session_id,
            )

            for entry in new_entries:
                if self._embedding_provider and not entry.embedding:
                    try:
                        emb = await self._embedding_provider.embed_query(entry.content)
                        entry.embedding = emb
                    except Exception as e:
                        logger.debug("Memory: embed failed: %s", e)
                self._store.add(entry)

            if new_entries:
                logger.info(
                    "Memory: stored %d new memories from session %s",
                    len(new_entries),
                    session_id or "?",
                )

            return len(new_entries)
        except Exception as e:
            logger.debug("Memory: learn failed: %s", e)
            return 0

    # ── Consolidation / Dreaming ──────────────────────────

    async def dream(
        self,
        model: Model,
        *,
        api_key: str | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Run memory consolidation ('dreaming').

        Automatically throttled to dream_interval_hours.
        Use force=True to bypass throttling (e.g., /dream command).
        """
        if not self._initialized:
            await self.initialize()
        if not self._consolidator:
            return None

        now = time.time()
        if not force:
            hours_since = (now - self._last_dream_time) / 3600
            if hours_since < self._config.dream_interval_hours:
                return None

        report = await self._consolidator.dream(model, api_key=api_key)
        self._last_dream_time = now

        return {
            "decayed": report.decayed,
            "strengthened": report.strengthened,
            "new_semantic": report.new_semantic,
            "merged": report.merged,
            "deleted": report.deleted,
            "cleaned": report.cleaned,
        }

    async def schedule_dreaming(
        self, model: Model, *, api_key: str | None = None
    ) -> None:
        """Start a background task for periodic dreaming."""
        if self._dream_task and not self._dream_task.done():
            return

        async def _dream_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(
                        self._config.dream_interval_hours * 3600
                    )
                    await self.dream(model, api_key=api_key)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Memory dream error: %s", e)
                    await asyncio.sleep(300)

        self._dream_task = asyncio.create_task(_dream_loop())

    # ── Direct Access ─────────────────────────────────────

    def get_store(self) -> MemoryStore | None:
        return self._store

    async def get_stats(self) -> dict[str, Any]:
        if not self._store:
            return {"enabled": False}
        stats = self._store.get_stats()
        stats["enabled"] = True
        stats["config"] = {
            "extract_interval": self._config.extract_interval_turns,
            "dream_interval_hours": self._config.dream_interval_hours,
            "max_context_tokens": self._config.max_context_tokens,
        }
        return stats

    async def list_memories(
        self,
        *,
        memory_type: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """List or search stored memories for human inspection (/memory command).

        - query given  → keyword search, ranked by relevance
        - memory_type   → filter to semantic/episodic/procedural
        - neither       → most important/recent first
        """
        if not self._initialized:
            await self.initialize()
        if not self._store:
            return []

        if query:
            keywords = {k for k in query.lower().split() if k}
            if not keywords:
                return []
            pairs = self._store.search_by_keywords(keywords, top_k=limit)
            return [entry for entry, _score in pairs]

        mt: MemoryType | None = None
        if memory_type:
            try:
                mt = MemoryType(memory_type)
            except ValueError:
                mt = None
        return self._store.list_all(memory_type=mt, limit=limit)

    async def clear_all(self) -> int:
        if not self._store:
            return 0
        # Delete everything, including sub-threshold (strength < 0.1) entries
        # that get_stats() would not count — return the true deleted count.
        entries = self._store.list_all(min_strength=0.0, limit=1_000_000)
        for entry in entries:
            self._store.delete(entry.id)
        return len(entries)

    async def close(self) -> None:
        if self._dream_task and not self._dream_task.done():
            self._dream_task.cancel()
            try:
                await self._dream_task
            except asyncio.CancelledError:
                pass
        if self._store:
            self._store.close()
