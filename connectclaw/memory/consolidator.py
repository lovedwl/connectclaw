"""Memory consolidation — the 'dreaming' process.

Reviews episodic memories and:
- Extracts patterns -> semantic memories
- Merges similar memories
- Decays unused memories
- Forgets irrelevant memories
- Strengthens frequently accessed memories
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from connectclaw.logging import get_logger
from connectclaw.provider.stream import stream_simple
from connectclaw.provider.types import Context, Model, UserMessage

from .prompts import (
    CONSOLIDATION_PROMPT,
    CONSOLIDATION_SYSTEM_PROMPT,
    CURATION_PROMPT,
    CURATION_SYSTEM_PROMPT,
)
from .store import MemoryStore
from .types import MemoryEntry, MemoryType

logger = get_logger(__name__)


@dataclass
class ConsolidationConfig:
    decay_halflife_days: float = 30.0
    decay_min_strength: float = 0.05
    access_boost: float = 0.05
    max_episodic_age_days: float = 90.0
    min_episodes_for_dream: int = 5
    dream_batch_size: int = 30
    # 整理（curation）：同主题冲突/过时/环境可读的事实，都在这一相处理
    curation_enabled: bool = True
    curation_batch_size: int = 80


@dataclass
class ConsolidationReport:
    decayed: int = 0
    strengthened: int = 0
    new_semantic: int = 0
    merged: int = 0
    deleted: int = 0
    cleaned: int = 0
    curated: int = 0   # 被更正/合并的条目数
    purged: int = 0     # 被清掉的过时条目数


class MemoryConsolidator:
    """Handles memory maintenance — dreaming, decay, merging."""

    def __init__(self, store: MemoryStore, config: ConsolidationConfig | None = None):
        self._store = store
        self._config = config or ConsolidationConfig()

    async def dream(
        self,
        model: Model,
        *,
        api_key: str | None = None,
        env_facts: str = "",
    ) -> ConsolidationReport:
        """Run a full consolidation cycle — 'dreaming'.

        Steps:
        1. Apply time-based decay to all memories
        2. Boost frequently accessed memories
        3. Consolidate old episodic memories (LLM call)
        4. Curate: resolve conflicts / drop stale / verify against env facts (LLM call)
        5. Cleanup memories below strength threshold
        """
        report = ConsolidationReport()

        report.decayed = self._apply_decay()
        logger.info("Dream: decayed %d memories", report.decayed)

        report.strengthened = self._boost_accessed()
        logger.info("Dream: strengthened %d memories", report.strengthened)

        episodes = self._store.list_all(
            memory_type=MemoryType.EPISODIC,
            min_strength=self._config.decay_min_strength,
        )
        old_episodes = [
            e
            for e in episodes
            if (time.time() - e.created_at) / 86400
            > self._config.max_episodic_age_days
        ]

        if len(old_episodes) >= self._config.min_episodes_for_dream:
            batch = old_episodes[: self._config.dream_batch_size]
            consolidation = await self._consolidate_episodes(
                batch, model, api_key=api_key
            )
            report.new_semantic = len(consolidation.get("new_semantic", []))
            report.merged = sum(
                max(0, len(g.get("memory_ids", [])) - 1)
                for g in consolidation.get("merge_groups", [])
            )
            report.deleted = len(consolidation.get("forget", []))
            logger.info(
                "Dream: created %d semantic, merged %d, deleted %d",
                report.new_semantic,
                report.merged,
                report.deleted,
            )

        # 整理相：解决同主题冲突、清理过时、按环境事实校验真伪、合并重复。
        # 用户 2026-09-24 拍板：这些属于"后台思考"该做的事，而不是写记忆时猜。
        if self._config.curation_enabled and model is not None:
            curation = await self.curate(model, env_facts, api_key=api_key)
            report.curated = curation.get("curated", 0)
            report.purged = curation.get("purged", 0)
            logger.info("Dream: curated %d, purged %d", report.curated, report.purged)

        report.cleaned = self._store.cleanup(self._config.decay_min_strength)
        logger.info("Dream: cleaned %d forgotten memories", report.cleaned)

        return report

    def apply_decay_only(self) -> int:
        """Apply time-based decay without LLM calls. Lightweight maintenance."""
        return self._apply_decay()

    # ── Clustering-based dedup (procedural / semantic merge) ──

    def cluster_memories(
        self, entries: list[MemoryEntry], k: int
    ) -> list[list[MemoryEntry]]:
        """Group ``entries`` into ``k`` clusters by embedding similarity.

        Memories without an embedding are returned as singletons appended at
        the end. Uses the deterministic numpy KMeans in :mod:`clustering` so the
        result is stable and testable without an LLM.
        """
        from .clustering import kmeans

        with_vec = [e for e in entries if e.embedding]
        singletons = [e for e in entries if not e.embedding]

        if len(with_vec) <= k:
            return [[e] for e in with_vec] + [[e] for e in singletons]

        import numpy as np

        matrix = np.array([e.embedding for e in with_vec], dtype=np.float32)
        labels = kmeans(matrix, k)

        buckets: list[list[MemoryEntry]] = [[] for _ in range(int(labels.max()) + 1)]
        for entry, label in zip(with_vec, labels):
            buckets[int(label)].append(entry)
        return [b for b in buckets if b] + [[e] for e in singletons]

    def consolidate_by_clustering(
        self, entries: list[MemoryEntry], k: int
    ) -> int:
        """Cluster then merge within each cluster. Returns count merged away.

        Deterministic (no LLM): keeps the first entry of each cluster, folds the
        others' content into it, and deletes them. This is the fallback / test
        path; production can replace the in-cluster merge with an LLM call that
        rewrites a single consolidated memory from the cluster.
        """
        clusters = self.cluster_memories(entries, k)
        merged = 0
        now = time.time()
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            keep, rest = cluster[0], cluster[1:]
            extras = "; ".join(r.content for r in rest if r.content)
            if extras:
                keep.content = f"{keep.content} | {extras}" if keep.content else extras
            keep.strength = min(1.0, keep.strength + 0.05 * len(rest))
            keep.last_accessed = now
            self._store.update(keep)
            for r in rest:
                self._store.delete(r.id)
                merged += 1
        return merged

    # ── Internal ──────────────────────────────────────────

    def _apply_decay(self) -> int:
        """Apply exponential decay to all memories based on age."""
        now = time.time()
        halflife_seconds = self._config.decay_halflife_days * 86400
        count = 0

        all_memories = self._store.list_all(min_strength=0.0)
        for entry in all_memories:
            # Decay by the time elapsed since the LAST dream, anchored on
            # last_decayed_at — NOT the full age since last_accessed. Applying
            # the full-age factor on every dream decays old memories ~30x
            # faster than the halflife intends (2^(-60d) re-multiplied daily).
            anchor = entry.last_decayed_at or entry.last_accessed
            elapsed = now - anchor
            if elapsed <= 0:
                continue

            decay_factor = 2 ** (-elapsed / halflife_seconds)
            new_strength = entry.strength * decay_factor

            floor = entry.importance * 0.3
            new_strength = max(new_strength, floor)

            changed = abs(new_strength - entry.strength) > 0.001
            if changed:
                entry.strength = new_strength
            # Persist the anchor even when the strength hit its floor —
            # otherwise the next dream would re-apply the whole elapsed span.
            entry.last_decayed_at = now
            self._store.update(entry)
            if changed:
                count += 1

        return count

    def _boost_accessed(self) -> int:
        """Strengthen memories that have been accessed recently."""
        count = 0

        all_memories = self._store.list_all(min_strength=0.1)
        for entry in all_memories:
            if entry.access_count == 0:
                continue

            boost = min(
                entry.access_count * self._config.access_boost,
                0.5,
            )
            new_strength = min(1.0, entry.strength + boost)

            if new_strength > entry.strength:
                entry.strength = new_strength
                self._store.update(entry)
                count += 1

            if entry.access_count > 10:
                entry.access_count = 5
                self._store.update(entry)

        return count

    async def _consolidate_episodes(
        self,
        episodes: list[MemoryEntry],
        model: Model,
        *,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Use LLM to consolidate old episodic memories into semantic knowledge."""
        existing_semantic = self._store.list_all(
            memory_type=MemoryType.SEMANTIC,
            min_strength=0.1,
            limit=50,
        )

        episodes_text = "\n".join(
            f"- [id={e.id}] {e.content}"
            + (f"\n  Detail: {e.detail}" if e.detail else "")
            for e in episodes
        )
        semantic_text = (
            "\n".join(f"- [id={e.id}] {e.content}" for e in existing_semantic)
            or "(no existing semantic memories)"
        )

        prompt_text = CONSOLIDATION_PROMPT.format(
            episodic_memories=episodes_text,
            existing_semantic=semantic_text,
        )

        context = Context(
            system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
            messages=[
                UserMessage(content=prompt_text, timestamp=time.time() * 1000)
            ],
        )

        text = await _call_llm(context, model, api_key=api_key)
        if not text:
            return {}

        result = _parse_json(text)
        self._apply_consolidation(result, episodes)
        return result

    async def curate(
        self,
        model: Model,
        env_facts: str = "",
        *,
        api_key: str | None = None,
    ) -> dict[str, int]:
        """整理记忆：同主题冲突、过时条目、环境可读的事实、重复条目。

        老账（2026-09-24 实测）：记忆库里「RAG已启用」被注入 84 次、而当天写的更正
        只出现 1 次——同主题的旧说法与新说法以同等权威并存。这类治理放在**后台做梦**
        里做：读全量记忆 + 环境事实快照，让模型给出 更正/遗忘/合并，再落库。

        返回 ``{"curated": n, "purged": n}``。
        """
        if model is None:
            # 没有可用模型时（例如"无 LLM 的做梦"）跳过整理相，不影响其它几相。
            return {"curated": 0, "purged": 0}

        memories = self._store.list_all(
            min_strength=self._config.decay_min_strength,
            limit=self._config.curation_batch_size,
        )
        if not memories:
            return {"curated": 0, "purged": 0}

        memories_text = "\n".join(
            f"- [id={e.id}] {_stamp(e)} {e.content}"
            + (f"\n  Detail: {e.detail}" if e.detail else "")
            for e in memories
        )
        prompt_text = CURATION_PROMPT.format(
            memories=memories_text,
            env_facts=env_facts.strip() or "(未提供环境快照)",
        )
        context = Context(
            system_prompt=CURATION_SYSTEM_PROMPT,
            messages=[UserMessage(content=prompt_text, timestamp=time.time() * 1000)],
        )

        text = await _call_llm(context, model, api_key=api_key)
        if not text:
            logger.debug("Dream: curation 无输出（模型未返回内容）")
            return {"curated": 0, "purged": 0}

        result = _parse_json(text)
        if not result:
            logger.warning("Dream: curation 输出无法解析为 JSON，跳过")
            return {"curated": 0, "purged": 0}
        return self._apply_curation(result, memories)

    def _apply_curation(
        self, result: dict[str, Any], memories: list[MemoryEntry]
    ) -> dict[str, int]:
        """把整理决定落库。只认库里真实存在的 id，且绝不编造内容。"""
        known = {e.id for e in memories}
        now = time.time()
        curated = purged = 0

        for item in result.get("update", []) or []:
            mid = str(item.get("id", ""))
            content = str(item.get("content", "")).strip()
            if mid not in known or not content:
                continue
            entry = self._store.get(mid)
            if not entry or entry.content == content:
                continue
            entry.content = content
            if "detail" in item and item.get("detail"):
                entry.detail = str(item["detail"])
            entry.last_accessed = now
            self._store.update(entry)
            curated += 1

        for group in result.get("merge_groups", []) or []:
            ids = [str(i) for i in (group.get("memory_ids") or []) if str(i) in known]
            merged_content = str(group.get("merged_content", "")).strip()
            if len(ids) < 2 or not merged_content:
                continue
            first = self._store.get(ids[0])
            if not first:
                continue
            first.content = merged_content
            first.last_accessed = now
            self._store.update(first)
            for mid in ids[1:]:
                self._store.delete(mid)
            curated += 1

        for item in result.get("new_semantic", []) or []:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            self._store.add(MemoryEntry(
                type=MemoryType.SEMANTIC,
                content=content,
                category=str(item.get("category", "") or ""),
                importance=min(1.0, max(0.0, float(item.get("importance", 0.6) or 0.6))),
                created_at=now,
                last_accessed=now,
                metadata={"curated": True, "reason": str(item.get("reason", ""))[:200]},
            ))
            curated += 1

        for item in result.get("forget", []) or []:
            # 兼容两种写法：字符串 id 或 {"id":..,"reason":..}
            mid = str(item.get("id", "")) if isinstance(item, dict) else str(item)
            if mid not in known:
                continue
            if self._store.delete(mid):
                purged += 1

        return {"curated": curated, "purged": purged}

    def _apply_consolidation(
        self, result: dict[str, Any], episodes: list[MemoryEntry]
    ) -> None:
        """Apply consolidation decisions to the store."""
        now = time.time()

        for item in result.get("new_semantic", []):
            content = item.get("content", "").strip()
            if not content:
                continue
            self._store.add(
                MemoryEntry(
                    type=MemoryType.SEMANTIC,
                    content=content,
                    category=item.get("category", ""),
                    importance=min(
                        1.0, max(0.0, float(item.get("importance", 0.6)))
                    ),
                    created_at=now,
                    last_accessed=now,
                    metadata={
                        "consolidated_from": item.get("source_episodes", [])
                    },
                )
            )

        for group in result.get("merge_groups", []):
            ids = group.get("memory_ids", [])
            if len(ids) < 2:
                continue
            merged_content = group.get("merged_content", "")
            merged_detail = group.get("merged_detail")
            if not merged_content:
                continue

            first = self._store.get(ids[0])
            if first:
                first.content = merged_content
                if merged_detail:
                    first.detail = merged_detail
                first.last_accessed = now
                first.strength = min(1.0, first.strength + 0.1)
                self._store.update(first)

            for mid in ids[1:]:
                self._store.delete(mid)

        for mid in result.get("strengthen", []):
            entry = self._store.get(mid)
            if entry:
                entry.strength = min(1.0, entry.strength + 0.1)
                entry.last_accessed = now
                self._store.update(entry)

        for mid in result.get("forget", []):
            self._store.delete(mid)

        for ep in episodes:
            ep.strength *= 0.7
            self._store.update(ep)


async def _call_llm(
    context: Context,
    model: Model,
    *,
    api_key: str | None = None,
) -> str:
    """Make a simple LLM call and return text."""
    parts: list[str] = []
    final: str | None = None
    async for event in stream_simple(model, context, api_key=api_key):
        if event.type == "text_delta" and event.delta:
            parts.append(event.delta)
        elif event.type == "done" and event.message:
            texts = [
                b.get("text", "")
                for b in event.message.content
                if b.get("type") == "text"
            ]
            final = "\n".join(texts)
            # break (not return) so the stream generator closes cleanly while
            # the event loop is still alive — avoids GeneratorExit at shutdown.
            break
    return final if final is not None else "".join(parts)



def _stamp(entry: MemoryEntry) -> str:
    """``[2026-07-27 · 0.31]`` —— 给 curation 的每条记忆带上时间与强度（判新旧用）。"""
    try:
        day = time.strftime("%Y-%m-%d", time.localtime(entry.created_at)) if entry.created_at else "?"
    except (OverflowError, OSError, ValueError):
        day = "?"
    strength = entry.strength if entry.strength is not None else 1.0
    return f"[{day} · {strength:.2f}]"

def _parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling markdown blocks and extra text."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}
