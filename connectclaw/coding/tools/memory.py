"""Memory tool — lets the agent inspect and retire its own memories.

Two actions:
- ``search``: read-only lookup so the agent can verify whether a memory is
  stale before proposing to forget it.
- ``forget``: soft-retire (strength → 0). The memory drops out of recall and
  is reclaimed by the next dream/cleanup cycle. Reversible until that cycle
  runs, so the model can mark stale data without irrevocably deleting user
  data.

Persona-grade memories (high-importance semantic — identity, tone, standing
preferences) are protected: ``forget`` skips them and reports the skip count,
so the user is nudged to remove those explicitly via ``/forget id <id>``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult


class MemoryTool(AgentTool):
    name = "memory"
    label = "memory"
    description = (
        "Inspect or retire long-term memories. "
        "action='search' looks up memories by keyword (read-only). "
        "action='forget' soft-retires memories matching a keyword — they stop "
        "being recalled and are cleaned up on the next consolidation cycle. "
        "Use 'forget' when the user says a memory is outdated or wrong. "
        "High-importance identity memories are protected from 'forget'; tell "
        "the user to remove those with /forget id <id>."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "forget"],
                "description": "search = look up memories; forget = soft-retire stale ones",
            },
            "keyword": {
                "type": "string",
                "description": "Keyword to search or forget by. Matched against memory content.",
            },
        },
        "required": ["action", "keyword"],
    }

    def __init__(self, memory):
        # MemorySubsystem — lazy-initialized, best-effort. We hold the ref but
        # every call re-checks .enabled so a disabled subsystem no-ops cleanly.
        self._memory = memory

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        if not getattr(self._memory, "enabled", False):
            return AgentToolResult(content=[{
                "type": "text",
                "text": "记忆系统已禁用。",
            }])

        action = params.get("action", "").strip()
        keyword = (params.get("keyword") or "").strip()
        if not keyword:
            return AgentToolResult(content=[{
                "type": "text",
                "text": "keyword 为必填项。",
            }])

        if action == "search":
            return self._search(keyword)
        if action == "forget":
            return await self._forget(keyword)
        return AgentToolResult(content=[{
            "type": "text",
            "text": f"未知操作：{action!r}。请使用 'search' 或 'forget'。",
        }])

    # ── actions ───────────────────────────────────────────

    def _search(self, keyword: str) -> AgentToolResult:
        entries = self._memory._find_by_keyword(keyword)  # noqa: SLF001 — same package
        if not entries:
            return AgentToolResult(content=[{
                "type": "text",
                "text": f"没有匹配 {keyword!r} 的记忆。",
            }])
        lines = [f"找到 {len(entries)} 条匹配 {keyword!r} 的记忆："]
        for e in entries[:30]:
            lines.append(
                f"- id={e.id} type={e.type.value} importance={e.importance:.2f} "
                f"strength={e.strength:.2f} :: {e.content}"
            )
        return AgentToolResult(content=[{"type": "text", "text": "\n".join(lines)}])

    async def _forget(self, keyword: str) -> AgentToolResult:
        # Count persona-protected ones up front so we can tell the model why
        # some matches were skipped.
        candidates = self._memory._find_by_keyword(keyword)  # noqa: SLF001
        protected = sum(1 for e in candidates if self._memory._is_persona(e))  # noqa: SLF001
        softened = await self._memory.soften_by_keyword(keyword)
        parts = [f"已软退役 {softened} 条匹配 {keyword!r} 的记忆。"]
        if protected:
            parts.append(
                f"{protected} 条 persona 级记忆受到保护 —— "
                "请用户使用 /forget id <id> 移除它们。"
            )
        parts.append("已退役的记忆将不再被召回，并会在下一个 /dream 周期中清理。")
        return AgentToolResult(content=[{"type": "text", "text": " ".join(parts)}])
