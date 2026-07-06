"""Memory types for ConnectClaw — layered memory system.

Three memory types mirroring human cognition:
- Semantic: stable facts, preferences, knowledge
- Episodic: specific events, decisions, conversation snippets
- Procedural: learned patterns, workflows, habits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry:
    id: str = ""
    type: MemoryType = MemoryType.SEMANTIC
    content: str = ""
    detail: str | None = None
    category: str = ""
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    embedding: list[float] = field(default_factory=list)
    source_session: str | None = None
    strength: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    entry: MemoryEntry
    score: float
    detail_level: str = "summary"
