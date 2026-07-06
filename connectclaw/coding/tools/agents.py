"""
Meta-tool `agents` — the single entry point for the sub-agent fleet.

WHY a meta-tool (not one tool per agent):
  - Immediacy: a directly-exposed tool's schema is frozen into the request at
    the start of a turn (agent_loop pins `context.tools` once). Adding a tool
    needs a new tools array → next turn. This tool's schema is CONSTANT; its
    targets are resolved *inside* `execute()` by re-scanning the agents dir.
    So an agent `create`d mid-turn is runnable in the very next tool call of
    the SAME turn.
  - Prefix cache: because named agents no longer each occupy a top-level tool,
    the main agent's tools array stays byte-stable across turns (only changes
    when config changes), extending the project's system-prompt cache
    discipline to the tools array. The volatile agent/tool catalog rides in the
    USER MESSAGE instead (see build_catalog), never the system prompt.

Actions (single `action` param):
  - list     — re-scan agents dir + list grantable tools (discovery)
  - describe — full instructions + tools of one agent
  - run      — run one agent (`agent`+`prompt`) or a DAG (`tasks[]` +
               `depends_on`); agents resolved at call time. Absorbs the old
               `task` tool.
  - create   — write ~/.connectclaw/agents/<name>.md. Absorbs the old
               `create_agent` tool. Runnable same-turn (run re-scans).

Design note: the DAG engine (layered topological scheduling + fleet card via
`on_update`) is migrated verbatim from the retired `task` tool; the only change
is that agents are resolved from a fresh re-scan rather than a per-turn-frozen
roster.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult, ThinkingLevel
from connectclaw.coding.tools.dynamic import load_dynamic_tools
from connectclaw.coding.tools.named_agents import (
    _NAME_RE,
    load_named_agents,
    parse_agent_file,
    render_agent_md,
)
from connectclaw.coding.tools.subagent import run_subagent
from connectclaw.provider.types import Model

# Default system prompt for an ad-hoc sub-agent (no named identity).
TASK_SYSTEM_PROMPT = """You are a focused sub-agent. Complete the assigned task and return results.

## Rules
- Only use the tools you've been given
- Be concise — return results, not conversation
- If you can't complete the task, explain why
- Use the bash tool for commands, read for files
- When done, output a clear summary of findings
"""

# How much of a dependency's output to inject into a dependent's prompt.
_DEP_INJECT_LIMIT = 2000


def _resolve_tools(names: list[str], available: list[AgentTool]) -> list[AgentTool]:
    """Resolve tool names to AgentTool instances."""
    by_name = {t.name: t for t in available}
    return [by_name[n] for n in names if n in by_name]


class AgentsTool(AgentTool):
    """Single entry point for the sub-agent fleet: list / describe / run / create."""

    name = "agents"
    label = "agents"
    description = (
        "Manage and run sub-agents through one entry point. Set `action`:\n"
        "- `list`: show available agents (to run) and grantable tools (to hand a "
        "new agent). Call this when unsure what exists.\n"
        "- `describe`: full instructions+tools of one agent — set `name`.\n"
        "- `run`: run ONE agent with `agent`+`prompt`, OR run many as a DAG with "
        "`tasks` (each task: `prompt`, optional `agent`/`tools`/`id`/`depends_on`; "
        "independent tasks run in parallel, a task's `depends_on` outputs are "
        "injected into its prompt). Live progress shows as a fleet card.\n"
        "- `create`: create/update a reusable agent — set `name`, `instructions` "
        "(its system prompt, natural language), optional `desc`/`tools`. Runnable "
        "immediately via `run` in the same turn.\n"
        "A newly created agent is NOT a separate tool — always reach it through "
        "`agents(action=\"run\", agent=\"<name>\")`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "describe", "run", "create"],
                "description": "Which operation to perform.",
            },
            "name": {
                "type": "string",
                "description": "describe: agent to inspect. create: new agent's name (letters/digits/_/-).",
            },
            "agent": {
                "type": "string",
                "description": "run (single): name of the agent to run, paired with `prompt`.",
            },
            "prompt": {
                "type": "string",
                "description": "run (single): the task to hand the agent.",
            },
            "tasks": {
                "type": "array",
                "description": "run (DAG): multiple sub-tasks. Independent ones run concurrently.",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The task for the sub-agent"},
                        "agent": {
                            "type": "string",
                            "description": "Optional: name of an existing agent to run this task (its identity + tools). If given, `tools` is ignored.",
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tool names for an ad-hoc sub-agent (default: ['read','bash'])",
                        },
                        "id": {"type": "string", "description": "Optional id, referenced by others' depends_on"},
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Ids that must finish first; their output is injected into this task's prompt",
                        },
                    },
                    "required": ["prompt"],
                },
            },
            "desc": {"type": "string", "description": "create: one-line description of the agent."},
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "create: tool names to grant the agent, e.g. ['read','bash'] (see action=list).",
            },
            "instructions": {
                "type": "string",
                "description": "create: the agent's system prompt — who it is, its job, output format (natural language).",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        model: Model,
        *,
        agents_dir: str,
        tools_dir: str,
        base_tools: list[AgentTool],
        cwd: str,
        api_key: str | None = None,
        thinking_level: ThinkingLevel = "off",
    ):
        self._model = model
        self._agents_dir = agents_dir
        self._tools_dir = tools_dir
        self._base_tools = list(base_tools)  # the fixed primitive instances
        self._cwd = cwd
        self._api_key = api_key
        self._thinking_level = thinking_level

    # ── Live re-scan (the immediacy mechanism) ──────────────

    def _grantable(self) -> list[AgentTool]:
        """The pool a sub-agent may be granted: base primitives + dynamic tools.

        Re-scanned every call so tools created this turn are grantable now.
        Deliberately NOT limited by the main agent's exposed-tool whitelist —
        a sub-agent can be granted tools the main agent itself doesn't hold.
        """
        return self._base_tools + load_dynamic_tools(self._tools_dir, self._cwd)

    def _named(self, grantable: list[AgentTool]) -> list[AgentTool]:
        """Re-scan named agents at call time (immediacy: create → run same turn)."""
        return load_named_agents(
            self._agents_dir,
            grantable,
            self._model,
            api_key=self._api_key,
            thinking_level=self._thinking_level,  # type: ignore[arg-type]
        )

    # ── Catalog (injected into the user message each turn) ──

    def build_catalog(self) -> str:
        """Two sections: runnable agents + grantable tools. Volatile → user msg."""
        agent_lines: list[str] = []
        p = Path(self._agents_dir)
        if p.is_dir():
            for f in sorted(p.glob("*.md")):
                meta = parse_agent_file(f)
                if meta and meta["name"]:
                    agent_lines.append(f"- {meta['name']} — {meta['desc']}")

        tool_names = [t.name for t in self._grantable()]

        parts: list[str] = []
        if agent_lines:
            parts.append(
                "## 可运行的 agents(用 `agents(action=\"run\", agent=\"<名>\")` 调用)\n"
                + "\n".join(agent_lines)
            )
        else:
            parts.append(
                "## 可运行的 agents\n(暂无。用 `agents(action=\"create\", ...)` 造一个,当轮即可 run)"
            )
        parts.append(
            "## 可授权给子 agent 的工具(create 时写进 `tools:[]`)\n" + ", ".join(tool_names)
        )
        return "\n\n".join(parts)

    # ── Dispatch ────────────────────────────────────────────

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        action = (params.get("action") or "").strip()
        if action == "list":
            return AgentToolResult(content=[{"type": "text", "text": self.build_catalog()}])
        if action == "describe":
            return self._do_describe(params.get("name") or "")
        if action == "create":
            return await self._do_create(params)
        if action == "run":
            return await self._do_run(params, signal, on_update)
        return AgentToolResult(content=[{
            "type": "text",
            "text": f"Unknown action '{action}'. Use one of: list | describe | run | create.",
        }])

    # ── describe ────────────────────────────────────────────

    def _do_describe(self, name: str) -> AgentToolResult:
        name = name.strip()
        if not name:
            return AgentToolResult(content=[{"type": "text", "text": "describe needs `name`."}])
        meta = parse_agent_file(Path(self._agents_dir) / f"{name}.md")
        if not meta:
            return AgentToolResult(content=[{
                "type": "text", "text": f"Agent '{name}' not found. Use action=list to see what exists.",
            }])
        text = (
            f"# {meta['name']}\n"
            f"desc: {meta['desc']}\n"
            f"tools: {meta['tools'] or '(none)'}\n\n"
            f"## instructions\n{meta['instructions']}"
        )
        return AgentToolResult(content=[{"type": "text", "text": text}])

    # ── create (absorbs create_agent) ───────────────────────

    async def _do_create(self, params: dict[str, Any]) -> AgentToolResult:
        name = (params.get("name") or "").strip()
        if not name or not _NAME_RE.match(name):
            return AgentToolResult(content=[{
                "type": "text", "text": "Invalid agent name. Use letters, digits, '_' or '-' only.",
            }])
        instructions = (params.get("instructions") or "").strip()
        if not instructions:
            return AgentToolResult(content=[{
                "type": "text", "text": "instructions is required (the agent's system prompt).",
            }])
        desc = params.get("desc", "") or ""
        tools = params.get("tools") or []
        if isinstance(tools, str):
            tools = [tools]

        content = render_agent_md(name, desc, [str(t) for t in tools], instructions)
        try:
            Path(self._agents_dir).mkdir(parents=True, exist_ok=True)
            (Path(self._agents_dir) / f"{name}.md").write_text(content, encoding="utf-8")
        except OSError as e:
            return AgentToolResult(content=[{"type": "text", "text": f"Failed to write agent: {e}"}])

        return AgentToolResult(content=[{
            "type": "text",
            "text": (
                f"✅ 已创建 agent `{name}`。可**立即**用 "
                f"`agents(action=\"run\", agent=\"{name}\", prompt=...)` 调用(无需等下一轮)。"
            ),
        }])

    # ── run (absorbs the task tool's DAG engine) ────────────

    async def _do_run(
        self,
        params: dict[str, Any],
        signal: asyncio.Event | None,
        on_update: Any,
    ) -> AgentToolResult:
        # Accept single-agent shorthand OR a DAG.
        tasks: list[dict] = params.get("tasks") or []
        if not tasks:
            agent = (params.get("agent") or "").strip()
            prompt = params.get("prompt", "")
            if agent:
                tasks = [{"agent": agent, "prompt": prompt}]
            elif prompt:
                tasks = [{"prompt": prompt}]
            else:
                return AgentToolResult(content=[{
                    "type": "text",
                    "text": "run needs `agent`+`prompt` (single) or `tasks` (DAG).",
                }])

        # Resolve the fleet's tool/agent universe ONCE per call, from a fresh
        # re-scan — this is what makes a just-created agent runnable now.
        grantable = self._grantable()
        named = self._named(grantable)

        # Assign ids + validate dependency references up front.
        for i, t in enumerate(tasks):
            if not t.get("id"):
                t["id"] = f"task{i + 1}"
        id_to_task = {t["id"]: t for t in tasks}
        for t in tasks:
            for dep in t.get("depends_on", []) or []:
                if dep not in id_to_task:
                    return AgentToolResult(content=[{
                        "type": "text",
                        "text": f"Task '{t['id']}' depends on unknown task '{dep}'.",
                    }])

        # Per-call state (keep local — this tool is a singleton).
        fleet: dict[str, dict] = {
            t["id"]: {"name": _label(t), "status": "waiting", "step": 0, "action": "", "trace": ""}
            for t in tasks
        }
        outputs: dict[str, str] = {}
        results: dict[str, dict] = {}
        done_ids: set[str] = set()
        failed_ids: set[str] = set()

        def _snapshot() -> list[dict]:
            return [dict(fleet[t["id"]]) for t in tasks]

        stop = asyncio.Event()

        async def _flush() -> None:
            if on_update:
                await on_update(AgentToolResult(
                    content=[{"type": "text", "text": ""}],
                    details={"fleet": _snapshot()},
                ))

        async def _flusher() -> None:
            while not stop.is_set():
                await _flush()
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
            await _flush()  # final frame

        flusher = asyncio.create_task(_flusher())
        try:
            await self._run_dag(
                tasks, fleet, outputs, results,
                done_ids, failed_ids, signal, grantable, named,
            )
        finally:
            stop.set()
            await flusher

        return _aggregate(tasks, results)

    async def _run_dag(
        self,
        tasks: list[dict],
        fleet: dict[str, dict],
        outputs: dict[str, str],
        results: dict[str, dict],
        done_ids: set[str],
        failed_ids: set[str],
        signal: asyncio.Event | None,
        grantable: list[AgentTool],
        named: list[AgentTool],
    ) -> None:
        """Layered topological scheduling: each round runs all ready tasks."""
        while len(done_ids) < len(tasks):
            if signal and signal.is_set():
                break

            progressed = False
            for t in tasks:
                tid = t["id"]
                if tid in done_ids:
                    continue
                deps = t.get("depends_on", []) or []
                if not all(d in done_ids for d in deps):
                    continue
                if any(d in failed_ids for d in deps):
                    fleet[tid].update(status="error", action="依赖失败,跳过")
                    results[tid] = {
                        "name": _label(t), "output": "",
                        "error": "skipped: dependency failed", "steps": 0,
                    }
                    outputs[tid] = ""
                    done_ids.add(tid)
                    failed_ids.add(tid)
                    progressed = True

            ready = [
                t for t in tasks
                if t["id"] not in done_ids
                and all(d in done_ids for d in (t.get("depends_on", []) or []))
            ]

            if not ready:
                if progressed:
                    continue
                for t in tasks:
                    if t["id"] not in done_ids:
                        fleet[t["id"]].update(status="error", action="循环依赖")
                        results[t["id"]] = {
                            "name": _label(t), "output": "",
                            "error": "unresolved dependency cycle", "steps": 0,
                        }
                        done_ids.add(t["id"])
                        failed_ids.add(t["id"])
                break

            batch = await asyncio.gather(*[
                self._run_task(t, outputs, fleet, grantable, named) for t in ready
            ])
            for t, res in zip(ready, batch):
                tid = t["id"]
                results[tid] = res
                outputs[tid] = res.get("output", "")
                done_ids.add(tid)
                if res.get("error"):
                    failed_ids.add(tid)

    async def _run_task(
        self,
        task: dict,
        outputs: dict[str, str],
        fleet: dict[str, dict],
        grantable: list[AgentTool],
        named: list[AgentTool],
    ) -> dict:
        """Build the spec (named agent or ad-hoc), inject deps, run it."""
        tid = task["id"]

        def _on_progress(name: str, status: str, step: int, action: str, trace: str = "") -> None:
            fleet[tid].update(status=status, step=step, action=action, trace=trace)

        prompt = task.get("prompt", "")
        dep_blocks = []
        for d in task.get("depends_on", []) or []:
            out = outputs.get(d, "")
            if out:
                dep_blocks.append(f"## 依赖 [{d}] 的产出:\n{out[:_DEP_INJECT_LIMIT]}")
        if dep_blocks:
            prompt = "\n\n".join(dep_blocks) + "\n\n---\n\n" + prompt

        agent_name = task.get("agent")
        if agent_name:
            found = _find_named(agent_name, named)
            if found is None:
                fleet[tid].update(status="error", action="未找到 agent")
                return {"name": agent_name, "output": "",
                        "error": f"named agent '{agent_name}' not found", "steps": 0}
            spec = {
                "name": agent_name,
                "system_prompt": getattr(found, "instructions", TASK_SYSTEM_PROMPT),
                "tools": getattr(found, "agent_tools", []),
                "prompt": prompt,
            }
        else:
            tool_names = task.get("tools") or ["read", "bash"]
            tools = _resolve_tools(tool_names, grantable)
            if not tools:
                fleet[tid].update(status="error", action="无有效工具")
                return {"name": _label(task), "output": "",
                        "error": f"no valid tools: {tool_names}", "steps": 0}
            spec = {
                "name": _label(task),
                "system_prompt": TASK_SYSTEM_PROMPT,
                "tools": tools,
                "prompt": prompt,
            }

        return await run_subagent(
            spec,
            model=self._model,
            api_key=self._api_key,
            thinking_level=self._thinking_level,
            on_progress=_on_progress,
        )


# ── Module helpers (migrated from task.py) ──────────────────────


def _label(task: dict) -> str:
    return task.get("agent") or task.get("id") or "task"


def _find_named(name: str, named: list[AgentTool]) -> AgentTool | None:
    """Find a named agent (duck-typed: has `instructions`) in the fresh scan."""
    for t in named:
        if t.name == name and hasattr(t, "instructions"):
            return t
    return None


def _aggregate(tasks: list[dict], results: dict[str, dict]) -> AgentToolResult:
    parts = []
    success = 0
    for t in tasks:
        tid = t["id"]
        res = results.get(tid)
        if res is None:
            status, body = "SKIPPED", "(not run)"
        elif res.get("error"):
            status, body = "ERROR", res["error"]
        else:
            status, body = "OK", res.get("output", "(no output)")
            success += 1
        parts.append(f"## [{tid}] {_label(t)}: {status}\n{body}")
    header = f"Completed {success}/{len(tasks)} tasks:\n\n"
    return AgentToolResult(
        content=[{"type": "text", "text": header + "\n\n---\n\n".join(parts)}],
        details={"results": results, "success": success, "total": len(tasks)},
    )


def create_agents_tool(
    model: Model,
    *,
    agents_dir: str,
    tools_dir: str,
    base_tools: list[AgentTool],
    cwd: str,
    api_key: str | None = None,
    thinking_level: ThinkingLevel = "off",
) -> AgentsTool:
    return AgentsTool(
        model,
        agents_dir=agents_dir,
        tools_dir=tools_dir,
        base_tools=base_tools,
        cwd=cwd,
        api_key=api_key,
        thinking_level=thinking_level,
    )
