"""ConnectClaw memory system — layered memory for persistent context.

Three-layer memory mirroring human cognition:
- Semantic: stable facts, preferences, knowledge
- Episodic: specific events, decisions, conversation snippets
- Procedural: learned patterns, workflows, habits

Cache-friendly: memory is prepended to user message, not system prompt.
"""

from __future__ import annotations

from .subsystem import MemoryConfig, MemorySubsystem

__all__ = ["MemoryConfig", "MemorySubsystem"]
