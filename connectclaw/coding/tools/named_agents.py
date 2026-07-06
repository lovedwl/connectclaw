"""
Named agents — an agent is a `.md` file, not a JSON blob.

Each `~/.connectclaw/agents/<name>.md` defines a reusable sub-agent:

    ---
    name: reviewer
    desc: 代码审查专家
    tools: [read, hash_read, bash]
    ---
    你是一个挑剔的代码审查者。只看 diff,关注正确性、边界、错误处理。
    输出:问题清单,每条给 文件:行号 + 为什么。不寒暄。

The frontmatter is minimal structure (name / desc / tools); the body is the
system prompt in natural language — the part a language model is actually good
at writing and reading. Each file auto-registers as a tool named `<name>`:
calling it spawns that agent once. Files are re-scanned every turn, so a newly
created agent is callable next turn.

`create_agent` is the first-class way for the agent (or user) to author one —
its core parameter is `instructions` (natural language), not a shell template.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import yaml

from connectclaw.agent.types import AgentTool, AgentToolResult, ThinkingLevel
from connectclaw.coding.tools.subagent import run_subagent
from connectclaw.provider.types import Model

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# ── Parsing ─────────────────────────────────────────────────────


def _split_frontmatter(raw: str) -> tuple[str | None, str]:
    """Split `---\\n<yaml>\\n---\\n<body>`. Returns (yaml_or_None, body)."""
    s = raw.lstrip("﻿").lstrip()
    if not s.startswith("---"):
        return None, raw
    lines = s.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None, raw
    fm = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:])
    return fm, body


def parse_agent_file(path: str | Path) -> dict[str, Any] | None:
    """Parse a `.md` agent file. Returns {name, desc, tools, instructions} or None."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    fm, body = _split_frontmatter(raw)
    if fm is None:
        return None
    try:
        meta = yaml.safe_load(fm) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict):
        return None

    name = meta.get("name") or Path(path).stem
    desc = meta.get("desc") or meta.get("description") or f"Named agent: {name}"
    tools = meta.get("tools") or []
    if isinstance(tools, str):
        tools = [tools]
    elif not isinstance(tools, list):
        tools = []
    return {
        "name": str(name),
        "desc": str(desc),
        "tools": [str(x) for x in tools],
        "instructions": body.strip(),
    }


def _resolve_tools(names: list[str], available: list[AgentTool]) -> list[AgentTool]:
    by_name = {t.name: t for t in available}
    return [by_name[n] for n in names if n in by_name]


# ── NamedAgentTool ──────────────────────────────────────────────


class NamedAgentTool(AgentTool):
    """A `.md`-defined agent, callable as a tool. Invoking it spawns the agent."""

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        agent_tools: list[AgentTool],
        model: Model,
        *,
        api_key: str | None = None,
        thinking_level: ThinkingLevel = "off",
    ):
        self.name = name
        self.label = name
        self.description = description
        self.parameters = {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "要交给该 agent 的任务"},
            },
            "required": ["prompt"],
        }
        # Exposed so the task tool can orchestrate this agent in a DAG.
        self.instructions = instructions
        self.agent_tools = agent_tools
        self._model = model
        self._api_key = api_key
        self._thinking_level = thinking_level

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        prompt = params.get("prompt", "")
        res = await run_subagent(
            {
                "name": self.name,
                "system_prompt": self.instructions,
                "tools": self.agent_tools,
                "prompt": prompt,
            },
            model=self._model,
            api_key=self._api_key,
            thinking_level=self._thinking_level,
            signal=signal,
        )
        if res.get("error"):
            return AgentToolResult(
                content=[{"type": "text", "text": f"[{self.name} error] {res['error']}"}],
                details={"steps": res.get("steps", 0)},
            )
        return AgentToolResult(
            content=[{"type": "text", "text": res.get("output") or "(no output)"}],
            details={"steps": res.get("steps", 0)},
        )


def load_named_agents(
    agents_dir: str,
    base_tools: list[AgentTool],
    model: Model,
    *,
    api_key: str | None = None,
    thinking_level: ThinkingLevel = "off",
) -> list[AgentTool]:
    """Scan agents_dir for `*.md` and build a NamedAgentTool for each.

    Safe to call every turn. Referenced tools resolve only from base_tools
    (base + dynamic), never from other named agents — no recursive registration.
    """
    path = Path(agents_dir)
    if not path.is_dir():
        return []
    out: list[AgentTool] = []
    seen: set[str] = set()
    for f in sorted(path.glob("*.md")):
        meta = parse_agent_file(f)
        if not meta or not meta["name"] or meta["name"] in seen:
            continue
        if not _NAME_RE.match(meta["name"]):
            continue
        seen.add(meta["name"])
        out.append(NamedAgentTool(
            name=meta["name"],
            description=meta["desc"],
            instructions=meta["instructions"],
            agent_tools=_resolve_tools(meta["tools"], base_tools),
            model=model,
            api_key=api_key,
            thinking_level=thinking_level,
        ))
    return out


# ── CreateAgentTool ─────────────────────────────────────────────


def render_agent_md(name: str, desc: str, tools: list[str], instructions: str) -> str:
    """Render a canonical `.md` agent file (frontmatter + body)."""
    fm = yaml.safe_dump(
        {"name": name, "desc": desc, "tools": list(tools)},
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    return f"---\n{fm}\n---\n{instructions.strip()}\n"


class CreateAgentTool(AgentTool):
    """Create or update a named agent by writing `<name>.md`."""

    name = "create_agent"
    label = "create_agent"
    description = (
        "Create or update a reusable named agent. Writes "
        "~/.connectclaw/agents/<name>.md and becomes callable next turn (as a "
        "tool named <name>, or via task's `agent` field). The agent's identity "
        "lives in `instructions` (natural language) — no JSON/shell templates."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "agent 名(字母/数字/_/-,无空格),也是调用它的工具名",
            },
            "desc": {"type": "string", "description": "一句话描述这个 agent 干什么"},
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "该 agent 可用的工具名,如 ['read','bash']",
            },
            "instructions": {
                "type": "string",
                "description": "该 agent 的 system prompt:它是谁、职责、输出要求(自然语言)",
            },
        },
        "required": ["name", "instructions"],
    }

    def __init__(self, agents_dir: str):
        self._agents_dir = agents_dir

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        name = (params.get("name") or "").strip()
        if not name or not _NAME_RE.match(name):
            return AgentToolResult(content=[{
                "type": "text",
                "text": "Invalid agent name. Use letters, digits, '_' or '-' only.",
            }])
        desc = params.get("desc", "") or ""
        tools = params.get("tools") or []
        if isinstance(tools, str):
            tools = [tools]
        instructions = params.get("instructions", "") or ""
        if not instructions.strip():
            return AgentToolResult(content=[{
                "type": "text", "text": "instructions is required (the agent's system prompt).",
            }])

        content = render_agent_md(name, desc, [str(t) for t in tools], instructions)
        try:
            Path(self._agents_dir).mkdir(parents=True, exist_ok=True)
            (Path(self._agents_dir) / f"{name}.md").write_text(content, encoding="utf-8")
        except OSError as e:
            return AgentToolResult(content=[{"type": "text", "text": f"Failed to write agent: {e}"}])

        return AgentToolResult(content=[{
            "type": "text",
            "text": (
                f"✅ 已创建 agent `{name}`(下一轮生效)。"
                f"可直接作为工具 `{name}(prompt=...)` 调用,或在 task 里用 `agent: \"{name}\"`。"
            ),
        }])


def create_create_agent_tool(agents_dir: str) -> CreateAgentTool:
    return CreateAgentTool(agents_dir)
