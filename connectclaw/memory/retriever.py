"""Memory retriever with graded detail levels — fuzzy recall.

Recent + important memories → full detail (clear)
Distant or low-importance → summary only (fuzzy)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from connectclaw.logging import get_logger

from .store import MemoryStore
from .types import MemoryEntry, MemoryType, SearchResult

logger = get_logger(__name__)


@dataclass
class RetrievalConfig:
    max_context_tokens: int = 2000
    recency_threshold_days: int = 7
    recent_detail_top_k: int = 5
    distant_summary_top_k: int = 10
    semantic_weight: float = 0.5
    recency_weight: float = 0.25
    importance_weight: float = 0.15
    strength_weight: float = 0.1
    min_score: float = 0.2
    # Hard cosine-similarity gate for embedding retrieval. Below this, a memory
    # is irrelevant regardless of recency/importance/strength. Measured on
    # BGE-M3 (zh): relevant hits land 0.50–0.73, unrelated queries peak <0.45.
    min_similarity: float = 0.45


class MemoryRetriever:
    """Retrieves relevant memories with graded detail levels."""

    def __init__(self, store: MemoryStore, config: RetrievalConfig | None = None):
        self._store = store
        self._config = config or RetrievalConfig()

    async def retrieve(
        self,
        query: str,
        *,
        query_embedding: list[float] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant memories with appropriate detail levels."""
        if query_embedding:
            results = await self._retrieve_by_embedding(query_embedding)
        else:
            results = await self._retrieve_by_keywords(query)

        for r in results:
            self._store.touch(r.entry.id)

        return results

    async def retrieve_formatted(
        self,
        query: str,
        *,
        query_embedding: list[float] | None = None,
    ) -> str:
        """Retrieve and format memories for context injection.

        Returns a string for user message injection (NOT system prompt,
        to preserve prompt cache), or "" if nothing relevant.
        """
        results = await self.retrieve(query, query_embedding=query_embedding)
        if not results:
            return ""

        return self._format_for_prompt(results)

    # ── Internal ──────────────────────────────────────────

    async def _retrieve_by_embedding(
        self, query_embedding: list[float]
    ) -> list[SearchResult]:
        raw = self._store.search_by_embedding(
            query_embedding,
            top_k=self._config.recent_detail_top_k
            + self._config.distant_summary_top_k
            + 10,
        )

        now = time.time()
        results: list[SearchResult] = []

        for entry, similarity in raw:
            if similarity < self._config.min_similarity:
                continue

            score = self._compute_score(entry, similarity, now)
            if score < self._config.min_score:
                continue

            detail_level = self._decide_detail_level(entry, now)
            results.append(
                SearchResult(
                    entry=entry,
                    score=score,
                    detail_level=detail_level,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return self._apply_budget(results)

    async def _retrieve_by_keywords(self, query: str) -> list[SearchResult]:
        keywords = set(query.lower().split())
        if not keywords:
            return []

        raw = self._store.search_by_keywords(keywords, top_k=30)

        now = time.time()
        results: list[SearchResult] = []

        for entry, keyword_score in raw:
            score = self._compute_score(entry, keyword_score, now)
            if score < self._config.min_score:
                continue

            detail_level = self._decide_detail_level(entry, now)
            results.append(
                SearchResult(
                    entry=entry,
                    score=score,
                    detail_level=detail_level,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return self._apply_budget(results)

    def _compute_score(
        self, entry: MemoryEntry, similarity: float, now: float
    ) -> float:
        """Multi-signal scoring: similarity + recency + importance + strength."""
        c = self._config

        age_days = (now - entry.last_accessed) / 86400
        recency = 1.0 / (1.0 + age_days / 30)

        score = (
            c.semantic_weight * similarity
            + c.recency_weight * recency
            + c.importance_weight * entry.importance
            + c.strength_weight * entry.strength
        )
        return score

    def _decide_detail_level(self, entry: MemoryEntry, now: float) -> str:
        """Decide whether to show full detail or just summary."""
        age_days = (now - entry.created_at) / 86400
        is_recent = age_days < self._config.recency_threshold_days
        is_important = entry.importance >= 0.7
        is_strong = entry.strength >= 0.5

        if is_recent and (is_important or is_strong):
            return "full"
        return "summary"

    def _apply_budget(self, results: list[SearchResult]) -> list[SearchResult]:
        """Apply token budget — keep top results within budget."""
        if not results:
            return []

        budget = self._config.max_context_tokens
        used = 0
        kept: list[SearchResult] = []

        for r in results:
            text = r.entry.content
            if r.detail_level == "full" and r.entry.detail:
                text += " " + r.entry.detail
            tokens = max(1, len(text) // 3)

            if used + tokens > budget:
                if r.detail_level == "full":
                    r.detail_level = "summary"
                    tokens = max(1, len(r.entry.content) // 3)
                    if used + tokens > budget:
                        break

            kept.append(r)
            used += tokens

        return kept

    def _format_for_prompt(self, results: list[SearchResult]) -> str:
        """Format search results for context injection."""
        if not results:
            return ""

        lines = ["<remembered-context>", "(Things you know from past interactions)"]

        for r in results:
            prefix = ""
            if r.entry.type == MemoryType.EPISODIC:
                prefix = "[past] "
            elif r.entry.type == MemoryType.PROCEDURAL:
                prefix = "[pattern] "

            if r.detail_level == "full" and r.entry.detail:
                lines.append(f"- {prefix}{r.entry.content}")
                lines.append(f"  Detail: {r.entry.detail}")
            else:
                lines.append(f"- {prefix}{r.entry.content}")

        lines.append("</remembered-context>")
        return "\n".join(lines)
