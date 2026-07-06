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


@dataclass
class ConsolidationReport:
    decayed: int = 0
    strengthened: int = 0
    new_semantic: int = 0
    merged: int = 0
    deleted: int = 0
    cleaned: int = 0


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
    ) -> ConsolidationReport:
        """Run a full consolidation cycle — 'dreaming'.

        Steps:
        1. Apply time-based decay to all memories
        2. Boost frequently accessed memories
        3. Consolidate old episodic memories (LLM call)
        4. Cleanup memories below strength threshold
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

        report.cleaned = self._store.cleanup(self._config.decay_min_strength)
        logger.info("Dream: cleaned %d forgotten memories", report.cleaned)

        return report

    def apply_decay_only(self) -> int:
        """Apply time-based decay without LLM calls. Lightweight maintenance."""
        return self._apply_decay()

    # ── Internal ──────────────────────────────────────────

    def _apply_decay(self) -> int:
        """Apply exponential decay to all memories based on age."""
        now = time.time()
        halflife_seconds = self._config.decay_halflife_days * 86400
        count = 0

        all_memories = self._store.list_all(min_strength=0.0)
        for entry in all_memories:
            age = now - entry.last_accessed
            if age <= 0:
                continue

            decay_factor = 2 ** (-age / halflife_seconds)
            new_strength = entry.strength * decay_factor

            floor = entry.importance * 0.3
            new_strength = max(new_strength, floor)

            if abs(new_strength - entry.strength) > 0.001:
                entry.strength = new_strength
                self._store.update(entry)
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
