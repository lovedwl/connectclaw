"""Cross-cutting contextvars for tool execution.

Kept in a dependency-free module so any tool layer (scripted_tools, subagent,
named_agents, agents) can read/write them without import cycles.

- current_conversation: which chat a tool call belongs to — stateful tool
  sessions isolate per chat (login state never crosses chats).
- current_subagent: which sub-agent is running ("" on the main agent). Set by
  run_subagent so each fleet member gets its OWN stateful session (e.g. its own
  browser), giving real parallelism instead of contending on one shared session.
"""

from __future__ import annotations

import contextvars

current_conversation: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_conversation", default="default"
)
current_subagent: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_subagent", default=""
)
