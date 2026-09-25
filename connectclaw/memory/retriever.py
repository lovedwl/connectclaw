"""Memory retriever with graded detail levels — fuzzy recall.

Recent + important memories → full detail (clear)
Distant or low-importance → summary only (fuzzy)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from connectclaw.injection import (
    FULL_HEADER,
    INCREMENTAL_HEADER,
    render_memory_block,
)
from connectclaw.logging import get_logger

from .store import MemoryStore
from .types import MemoryEntry, MemoryType, SearchResult

logger = get_logger(__name__)


def _content_referenced(content: str, reply_lower: str, *, min_chars: int = 4) -> bool:
    """True if a distinctive substring of ``content`` appears in the reply.

    Normalizes by stripping whitespace and lowercasing, then checks whether any
    ``min_chars``-length contiguous slice of the content survives in the reply.
    Whitespace-stripping lets Chinese (no word boundaries) match on character
    runs, while latin identifiers still match as whole tokens. ``min_chars``
    avoids spurious boosts from tiny common fragments.
    """
    import re as _re

    norm_content = _re.sub(r"\s+", "", content).lower()
    norm_reply = _re.sub(r"\s+", "", reply_lower)
    if not norm_content:
        return False
    if len(norm_content) <= min_chars:
        return norm_content in norm_reply
    for i in range(len(norm_content) - min_chars + 1):
        if norm_content[i : i + min_chars] in norm_reply:
            return True
    return False


@dataclass
class RetrievalConfig:
    max_context_tokens: int = 2000
    recency_threshold_days: int = 7
    recent_detail_top_k: int = 3
    distant_summary_top_k: int = 5
    semantic_weight: float = 0.5
    recency_weight: float = 0.25
    importance_weight: float = 0.15
    strength_weight: float = 0.1
    min_score: float = 0.2
    # ── 召回"多余"的两道闸（2026-09-25 用户反馈"召回得有点多余"后加）──
    # 实测：本机 BGE 中文余弦挤在 0.5~0.85 的窄带里——无关项也有 0.52，而明显是
    # 同一件事的两种说法才 0.855。所以**绝对阈值卡不出边界**（min_similarity=0.48
    # 形同虚设，一轮能放进 15 条）。改成跟本轮最佳命中比：低于 top×keep_ratio 的丢掉。
    similarity_keep_ratio: float = 0.85
    # 每轮注入的记忆条数硬上限（原来由 recent+distant 决定的 15 条，太多）。
    recall_top_k: int = 8
    # Hard cosine-similarity gate for embedding retrieval. Below this, a memory
    # is irrelevant regardless of recency/importance/strength. Measured on
    # BGE-base-zh-v1.5 (zh, 2026-09): relevant hits land 0.50–0.55, unrelated
    # crosstalk peaks ~0.47 — 0.48 keeps real hits in and borderline noise out.
    min_similarity: float = 0.48
    # Persona injection: high-importance semantic memories (how to address the
    # user, tone, standing preferences) are injected EVERY turn, bypassing the
    # similarity gate — so identity is present from the first "hi", not only
    # when the user's message happens to match it. Set ABOVE the confirm_usage
    # auto-boost ceiling (0.65): nothing may climb into the always-on block on
    # its own — recall confirmations used to walk config-snapshot memories up
    # to 0.8, hijacking all persona slots with soon-stale "facts".
    persona_min_importance: float = 0.85
    persona_top_k: int = 8


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
        """Retrieve relevant memories with appropriate detail levels.

        Does NOT touch access stats — recall is not usage. Call
        :meth:`confirm_usage` after the assistant's reply is produced so that
        only memories actually reflected in the response get strengthened.
        """
        if query_embedding:
            results = await self._retrieve_by_embedding(
                query_embedding, query=query
            )
        else:
            results = await self._retrieve_by_keywords(query)
        return results

    def confirm_usage(
        self,
        reply_text: str,
        results: list[SearchResult],
        *,
        confirmed_out: list[str] | None = None,
    ) -> int:
        """Mark memories actually used in the reply as accessed.

        A memory counts as "used" if a distinctive fragment of its content
        appears in the reply. We match on the longest content token-run so a
        one-word hit doesn't count (avoid spurious boosts on common words),
        and persona memories (always injected) are confirmed too — they were
        honored simply by the reply existing. Returns the count confirmed.

        ``confirmed_out``: optional list receiving the confirmed memory ids
        (评估集用它拿逐条标签；返回值保持 int 兼容旧调用方).
        """
        if not reply_text or not results:
            return 0
        reply_lower = reply_text.lower()
        confirmed = 0
        for r in results:
            content = (r.entry.content or "").strip()
            if not content:
                continue
            # persona block (score==1.0) is always-injected; count it as used.
            if r.score == 1.0 or _content_referenced(content, reply_lower):
                self._store.touch(r.entry.id)
                if confirmed_out is not None:
                    confirmed_out.append(r.entry.id)
                # Auto-boost importance for memories confirmed as useful. The
                # 0.65 ceiling stays BELOW the persona threshold (0.85) — a
                # repeatedly-confirmed memory must get ranked higher, but never
                # promote itself into the every-turn persona block.
                if r.score < 1.0:  # skip persona (already trusted)
                    new_imp = min(0.65, r.entry.importance + 0.02)
                    if new_imp > r.entry.importance:
                        self._store.update_importance(r.entry.id, new_imp)
                        r.entry.importance = new_imp
                confirmed += 1
        return confirmed

    async def retrieve_formatted(
        self,
        query: str,
        *,
        query_embedding: list[float] | None = None,
    ) -> tuple[str, list[SearchResult]]:
        """Retrieve and format memories for context injection.

        Returns ``(formatted_text, results)``. The text is for user-message
        injection (NOT system prompt, to preserve prompt cache); ``results``
        is the list of recalled memories the caller should pass to
        :meth:`confirm_usage` after the reply is produced. Empty text if
        nothing relevant.
        """
        results = await self.retrieve(query, query_embedding=query_embedding)

        # Always-on persona block, merged ahead of per-query hits (deduped by id).
        persona = self._retrieve_persona()
        if persona:
            seen = {r.entry.id for r in persona}
            results = persona + [r for r in results if r.entry.id not in seen]

        if not results:
            return "", []

        return self._format_for_prompt(results), results

    # ── Internal ──────────────────────────────────────────

    def _retrieve_persona(self) -> list[SearchResult]:
        """Unconditional high-importance semantic memories (identity/persona).

        Shown at full detail and ranked first — these are the standing facts the
        assistant should always honor (how to address the user, tone).
        """
        entries = self._store.list_persona(
            min_importance=self._config.persona_min_importance,
            limit=self._config.persona_top_k,
        )
        out: list[SearchResult] = []
        for entry in entries:
            # score=1.0 marks persona — confirm_usage treats it as always-used.
            # Do NOT touch here; usage confirmation is centralized.
            out.append(
                SearchResult(entry=entry, score=1.0, detail_level="full")
            )
        return out

    async def _retrieve_by_embedding(
        self, query_embedding: list[float], *, query: str = ""
    ) -> list[SearchResult]:
        raw = self._store.search_by_embedding(
            query_embedding,
            top_k=self._config.recent_detail_top_k
                + self._config.distant_summary_top_k
                + 10,
        )

        # BM25 fusion: add a keyword signal so exact-term matches (names, IDs,
        # paths, error codes) surface even when embedding similarity is modest.
        bm25_scores: dict[str, float] = {}
        if query:
            try:
                from .bm25 import BM25Index

                corpus = [e.content or "" for e, _ in raw]
                idx = BM25Index(corpus)
                scores = idx.score(query)
                bm25_scores = {raw[i][0].id: scores[i] for i in range(len(raw))}
            except Exception:
                bm25_scores = {}

        now = time.time()
        results: list[SearchResult] = []
        sims: dict[str, float] = {}
        bm_hits: set[str] = set()

        for entry, similarity in raw:
            if similarity < self._config.min_similarity:
                continue

            bm = bm25_scores.get(entry.id, 0.0)
            # Normalize bm25 into [0,1] against the max in this batch so it
            # can be folded into the weighted score alongside similarity.
            max_bm = max(bm25_scores.values()) if bm25_scores else 0.0
            bm_norm = (bm / max_bm) if max_bm > 0 else 0.0

            score = self._compute_score(entry, similarity, now, bm25=bm_norm)
            if score < self._config.min_score:
                continue

            detail_level = self._decide_detail_level(entry, now)
            sims[entry.id] = similarity
            if bm_norm > 0:
                bm_hits.add(entry.id)
            results.append(
                SearchResult(
                    entry=entry,
                    score=score,
                    detail_level=detail_level,
                    similarity=similarity,
                    bm25=bm_norm,
                )
            )

        # 相对相关性下限：跟本轮最佳命中比，差太多的直接丢掉（BM25 精确命中豁免——
        # 那是特意融合进来的另一路信号，不该被相似度比值砍掉）。
        if results:
            top_sim = max(sims[r.entry.id] for r in results)
            cut = top_sim * self._config.similarity_keep_ratio
            kept = [r for r in results if sims[r.entry.id] >= cut or r.entry.id in bm_hits]
            if len(kept) < len(results):
                logger.debug(
                    "Memory: 相对相关性下限丢弃 %d 条（top=%.3f cut=%.3f）",
                    len(results) - len(kept), top_sim, cut,
                )
            results = kept

        results.sort(key=lambda r: r.score, reverse=True)
        results = self._apply_type_quota(results)
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
                    bm25=keyword_score,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        results = self._apply_type_quota(results)
        return self._apply_budget(results)

    # Per-type floor so one memory class can't monopolize the TopK
    # (e.g. a query that matches lots of procedural memories shouldn't
    # crowd out semantic ones). Guarantees each type a minimum number of
    # slots before filling the rest by score.
    _TYPE_QUOTA_FLOOR = {
        MemoryType.SEMANTIC: 2,
        MemoryType.EPISODIC: 1,
        MemoryType.PROCEDURAL: 1,
    }

    def _apply_type_quota(self, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        total = min(
            self._config.recent_detail_top_k + self._config.distant_summary_top_k,
            self._config.recall_top_k,
        )
        floors = self._TYPE_QUOTA_FLOOR
        kept: list[SearchResult] = []
        seen_by_type: dict[MemoryType, int] = {t: 0 for t in floors}

        # Pass 1: fill each type's floor from the score-sorted list.
        remaining = []
        for r in results:
            t = r.entry.type
            if t in floors and seen_by_type.get(t, 0) < floors[t]:
                kept.append(r)
                seen_by_type[t] = seen_by_type.get(t, 0) + 1
            else:
                remaining.append(r)

        # Pass 2: fill the rest by score until we reach the total budget.
        for r in remaining:
            if len(kept) >= total:
                break
            kept.append(r)

        kept.sort(key=lambda r: r.score, reverse=True)
        return kept

    def _compute_score(
        self, entry: MemoryEntry, similarity: float, now: float,
        *, bm25: float = 0.0,
    ) -> float:
        """Multi-signal scoring: similarity + bm25 + recency + importance + strength.
        Newer memories get a temporary freshness boost so they have a chance
        to be seen and confirmed before fading.
        """
        c = self._config

        age_days_recency = (now - entry.last_accessed) / 86400
        recency = 1.0 / (1.0 + age_days_recency / 30)

        # Embedding similarity and BM25 together carry the same "relevance"
        # budget; split it so a hit on either signal can surface the memory.
        relevance = max(similarity, bm25)
        score = (
            c.semantic_weight * relevance
            + c.recency_weight * recency
            + c.importance_weight * entry.importance
            + c.strength_weight * entry.strength
        )

        # Freshness boost: memories < 7 days old get up to +30% lift
        # so newly extracted info has a chance to be recalled and confirmed.
        age_days_created = (now - entry.created_at) / 86400
        if age_days_created < 7:
            boost = 1.0 + (7.0 - age_days_created) / 7.0 * 0.3
            score *= boost

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
        """Apply token budget and the per-turn recall cap."""
        if not results:
            return []

        # 条数硬上限：token 预算管不住"条数"，而一轮塞十几条记忆正是"召回多余"
        # 的直接来源。两条检索路径都收口到这里，所以闸门加在这一层。
        limit = self._config.recall_top_k
        if limit and len(results) > limit:
            logger.debug("Memory: 召回条数上限 %d，丢弃尾部 %d 条", limit, len(results) - limit)
            results = results[:limit]

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

    def render_item_lines(self, results: list[SearchResult]) -> list[tuple[str, str, str]]:
        """``[(memory_id, 渲染后的条目文本, 原始内容)]``。

        渲染文本是**增量比对**的单位：只要它没变，这一条就不必再注入一遍
        （按需注入，见 connectclaw/injection.py）。Detail 跟在同一条里（带换行），
        整条一起参与比对。
        """
        out: list[tuple[str, str, str]] = []
        for r in results:
            prefix = ""
            if r.entry.type == MemoryType.EPISODIC:
                prefix = "[past] "
            elif r.entry.type == MemoryType.PROCEDURAL:
                prefix = "[pattern] "

            line = f"- {_freshness_stamp(r.entry)}{prefix}{r.entry.content}"
            if r.detail_level == "full" and r.entry.detail:
                line += f"\n  Detail: {r.entry.detail}"
            out.append((r.entry.id, line, r.entry.content))
        return out

    def format_block(self, lines: list[str], *, incremental: bool = False) -> str:
        """把若干条目行包成一个注入块。文本格式只在 injection 里定义一份。"""
        return render_memory_block(
            lines, header=INCREMENTAL_HEADER if incremental else FULL_HEADER
        )

    def _format_for_prompt(self, results: list[SearchResult]) -> str:
        """完整块（首次注入 / 调试用）。

        每条带上 ``[日期 · 强度]``。库里本来就存着 created_at / strength /
        access_count，但过去一个字都不输出，于是同一主题的旧说法与新说法以
        同等权威并列（实测「RAG 已启用」被注入 84 次，当天写的更正只 1 次），
        模型没有任何依据判新旧，只能凭语序猜。日期给新旧，强度给可信度。
        """
        return self.format_block([line for _, line, _ in self.render_item_lines(results)])


def _freshness_stamp(entry: MemoryEntry) -> str:
    """``[2026-07-27 · 0.31] `` —— 记录日期与强度；拿不到时间就不显示（不编造）。"""
    if not entry.created_at:
        return ""
    try:
        day = time.strftime("%Y-%m-%d", time.localtime(entry.created_at))
    except (OverflowError, OSError, ValueError):
        return ""
    try:
        strength = 1.0 if entry.strength is None else float(entry.strength)
    except (TypeError, ValueError):
        strength = 1.0
    return f"[{day} · {strength:.2f}] "
