"""JSONL session persistence for ConnectClaw.

Mirrors pi-mono's session tree model with simplified entry types.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import aiofiles
import aiofiles.os


# ── Session Entries ────────────────────────────────────────────


@dataclass
class MessageEntry:
    type: Literal["message"] = "message"
    id: str = ""
    parent_id: str | None = None
    timestamp: str = ""
    message: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactionEntry:
    type: Literal["compaction"] = "compaction"
    id: str = ""
    parent_id: str | None = None
    timestamp: str = ""
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0


@dataclass
class ModelChangeEntry:
    type: Literal["model_change"] = "model_change"
    id: str = ""
    parent_id: str | None = None
    timestamp: str = ""
    provider: str = ""
    model_id: str = ""


@dataclass
class BranchSummaryEntry:
    type: Literal["branch_summary"] = "branch_summary"
    id: str = ""
    parent_id: str | None = None
    timestamp: str = ""
    summary: str = ""
    from_id: str = ""


SessionEntry = MessageEntry | CompactionEntry | ModelChangeEntry | BranchSummaryEntry


# ── Session Header ─────────────────────────────────────────────


@dataclass
class SessionHeader:
    type: Literal["session"] = "session"
    version: int = 3
    id: str = ""
    created_at: str = ""
    cwd: str = ""


# ── Session Context ────────────────────────────────────────────


@dataclass
class SessionContext:
    messages: list[dict[str, Any]] = field(default_factory=list)
    compaction_summary: str | None = None
    branch_summaries: list[str] = field(default_factory=list)


# ── Session Storage Backend ────────────────────────────────────


class JsonlSessionStorage:
    """JSONL file-based session storage."""

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._entries: list[SessionEntry] = []
        self._by_id: dict[str, SessionEntry] = {}
        self._current_leaf_id: str | None = None
        self._header: SessionHeader | None = None

    @classmethod
    async def open(cls, file_path: str) -> JsonlSessionStorage:
        """Parse existing JSONL file."""
        storage = cls(file_path)
        if not os.path.exists(file_path):
            return storage

        async with aiofiles.open(file_path, "r") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = data.get("type", "")
                if entry_type == "session":
                    storage._header = SessionHeader(**data)
                elif entry_type == "message":
                    storage._entries.append(MessageEntry(**data))
                elif entry_type == "compaction":
                    storage._entries.append(CompactionEntry(**data))
                elif entry_type == "model_change":
                    storage._entries.append(ModelChangeEntry(**data))
                elif entry_type == "branch_summary":
                    storage._entries.append(BranchSummaryEntry(**data))

        for entry in storage._entries:
            storage._by_id[entry.id] = entry

        # Determine leaf: last entry
        if storage._entries:
            storage._current_leaf_id = storage._entries[-1].id

        return storage

    @classmethod
    async def create(
        cls, file_path: str, cwd: str, session_id: str | None = None
    ) -> JsonlSessionStorage:
        """Create a new session JSONL file."""
        sid = session_id or str(uuid.uuid4()).replace("-", "")[:12]

        header = SessionHeader(
            id=sid,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cwd=cwd,
        )

        # Ensure directory exists
        await aiofiles.os.makedirs(os.path.dirname(file_path), exist_ok=True)

        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(header.__dict__) + "\n")

        storage = cls(file_path)
        storage._header = header
        return storage

    async def append_entry(self, entry: SessionEntry) -> None:
        """Append one line to JSONL file and update in-memory state."""
        entry_data = _entry_to_dict(entry)
        async with aiofiles.open(self._file_path, "a") as f:
            await f.write(json.dumps(entry_data, default=str) + "\n")

        self._entries.append(entry)
        self._by_id[entry.id] = entry
        self._current_leaf_id = entry.id

    async def append_message(self, message: dict[str, Any]) -> str:
        """Append a message entry, returns the entry ID."""
        entry_id = str(uuid.uuid4()).replace("-", "")[:12]
        entry = MessageEntry(
            id=entry_id,
            parent_id=self._current_leaf_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            message=message,
        )
        await self.append_entry(entry)
        return entry_id

    async def append_compaction(
        self, summary: str, first_kept_entry_id: str, tokens_before: int
    ) -> str:
        """Append a compaction entry."""
        entry_id = str(uuid.uuid4()).replace("-", "")[:12]
        entry = CompactionEntry(
            id=entry_id,
            parent_id=self._current_leaf_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
        )
        await self.append_entry(entry)
        return entry_id

    async def get_path_to_root(self, leaf_id: str | None = None) -> list[SessionEntry]:
        """Collect all entries from root to leaf, handling branching (parallel tool calls).

        The session DAG branches when an assistant message issues multiple parallel
        tool calls — each tool result shares the same parent.  A simple parent-chain
        walk would follow only one sibling and silently drop the others, breaking
        tool_call/tool_result pairing.

        This method does a full DFS traversal from the root node, visiting siblings
        in insertion order, so all parallel tool results are preserved.
        """
        lid = leaf_id or self._current_leaf_id
        if not lid:
            return []

        # Build children map keyed by parent_id.
        children: dict[str, list[SessionEntry]] = {}
        for entry in self._entries:
            pid: str | None = None
            if hasattr(entry, "parent_id"):
                pid = entry.parent_id
            if pid:
                children.setdefault(pid, []).append(entry)

        # Find roots (entries with no parent_id).
        roots = [
            e for e in self._entries
            if not (hasattr(e, "parent_id") and e.parent_id)
        ]

        result: list[SessionEntry] = []

        def walk(entry_id: str) -> None:
            entry = self._by_id.get(entry_id)
            if entry is None:
                return
            result.append(entry)
            # Visit children in insertion order (preserved by _entries list order).
            for child in children.get(entry_id, []):
                walk(child.id)

        for root in roots:
            walk(root.id)

        return result

    @property
    def entries(self) -> list[SessionEntry]:
        return list(self._entries)

    @property
    def session_id(self) -> str | None:
        return self._header.id if self._header else None


# ── Session Repository ─────────────────────────────────────────


class SessionRepo:
    """Manages session discovery, creation, and listing."""

    def __init__(self, sessions_dir: str):
        self._dir = sessions_dir

    async def ensure_dir(self) -> None:
        await aiofiles.os.makedirs(self._dir, exist_ok=True)

    def session_path(self, session_id: str) -> str:
        return os.path.join(self._dir, f"{session_id}.jsonl")

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with metadata."""
        await self.ensure_dir()
        sessions = []
        try:
            entries = os.listdir(self._dir)
        except FileNotFoundError:
            return []

        for fname in sorted(entries, reverse=True):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(self._dir, fname)
            try:
                storage = await JsonlSessionStorage.open(fpath)
                if storage._header:
                    sessions.append({
                        "id": storage._header.id,
                        "created_at": storage._header.created_at,
                        "cwd": storage._header.cwd,
                        "path": fpath,
                        "entry_count": len(storage.entries),
                    })
            except Exception:
                continue

        return sessions

    async def create_session(self, cwd: str, session_id: str | None = None) -> JsonlSessionStorage:
        """Create a new session."""
        await self.ensure_dir()
        sid = session_id or str(uuid.uuid4()).replace("-", "")[:12]
        fpath = self.session_path(sid)
        return await JsonlSessionStorage.create(fpath, cwd, sid)

    async def open_session(self, session_id: str) -> JsonlSessionStorage | None:
        """Open an existing session by ID."""
        fpath = self.session_path(session_id)
        if not os.path.exists(fpath):
            return None
        return await JsonlSessionStorage.open(fpath)


# ── Helpers ────────────────────────────────────────────────────


def _entry_to_dict(entry: SessionEntry) -> dict:
    """Convert entry to serializable dict."""
    if hasattr(entry, "__dict__"):
        return {k: v for k, v in entry.__dict__.items() if not k.startswith("_")}
    if isinstance(entry, dict):
        return dict(entry)
    return {"value": str(entry)}


def build_session_context(entries: list[SessionEntry]) -> SessionContext:
    """Walk path-to-root and reconstruct message list with compaction."""
    from connectclaw.provider.types import Message, normalize_message

    messages: list[Message] = []
    compaction_summary: str | None = None
    branch_summaries: list[str] = []

    for entry in entries:
        if entry.type == "message":
            msg = normalize_message(entry.message)
            messages.append(msg)
        elif entry.type == "compaction":
            compaction_summary = entry.summary
            messages = []
        elif entry.type == "branch_summary":
            branch_summaries.append(entry.summary)

    return SessionContext(
        messages=messages,
        compaction_summary=compaction_summary,
        branch_summaries=branch_summaries,
    )
