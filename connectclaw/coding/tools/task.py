"""
Task tool — orchestrate sub-agents as a DAG.

Each task item is spawned as a child agent. Items may declare `depends_on`
other items by `id`; dependencies run first and their output is injected into
the dependent's prompt. Items with no (unmet) dependency run concurrently.

An item may either:
  - name a `agent` (a `.md`-defined named agent) → run with that identity, or
  - give ad-hoc `prompt` + `tools` → run with the default sub-agent prompt.

Live progress is bubbled to the caller via `on_update` as a "fleet" snapshot
(one row per sub-agent), which the Feishu channel renders as a separate,
throttled fleet card.

Example:
  task(tasks=[
    {"id": "scan",  "prompt": "找出所有 flaky 测试", "tools": ["read", "bash"]},
    {"id": "fix",   "agent": "fixer", "prompt": "修复它们", "depends_on": ["scan"]},
  ])
"""

from __future__ import annotations

import asyncio
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult, ThinkingLevel
from connectclaw.coding.tools.subagent import run_subagent
from connectclaw.provider.types import Model

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


class TaskTool(AgentTool):
    """Spawn sub-agents as a DAG. Independent items run concurrently."""

    name = "task"
    label = "task"
    description = (
        "Spawn sub-agents to work on tasks, orchestrated as a DAG. "
        "Independent tasks run in parallel; a task with `depends_on` waits for "
        "its dependencies and receives their output in its prompt. "
        "Each task can either name an existing `agent` (a named agent you or the "
        "user created) or give an ad-hoc `prompt` + `tools`. "
        "Live progress is shown as a fleet card. Use for parallel or staged work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "Tasks to run. Independent ones run concurrently.",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The task for the sub-agent to complete",
                        },
                        "agent": {
                            "type": "string",
                            "description": (
                                "Optional: name of an existing named agent to run this "
                                "task with (uses its identity + tools). If given, `tools` "
                                "is ignored."
                            ),
                        },
                        "tools": {
                            "type": "array",
                            "description": "Tool names for an ad-hoc sub-agent (default: ['read', 'bash'])",
                            "items": {"type": "string"},
                        },
                        "id": {
                            "type": "string",
                            "description": "Optional id for this task, referenced by others' depends_on",
                        },
                        "depends_on": {
                            "type": "array",
                            "description": "Optional ids of tasks that must finish before this one; their output is injected into this task's prompt",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["prompt"],
                },
            },
        },
        "required": ["tasks"],
    }

    def __init__(
        self,
        model: Model,
        *,
        api_key: str | None = None,
        thinking_level: ThinkingLevel = "off",
        all_tools: list[AgentTool] | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._thinking_level = thinking_level
        self._all_tools = all_tools or []

    # ── Execution ───────────────────────────────────────────

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        tasks: list[dict] = params.get("tasks") or []
        if not tasks:
            return AgentToolResult(content=[{"type": "text", "text": "No tasks provided."}])

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

        # Per-call state (TaskTool is a singleton — keep everything local so
        # concurrent task calls never clobber each other).
        fleet: dict[str, dict] = {
            t["id"]: {"name": self._label(t), "status": "waiting", "step": 0, "action": "", "trace": ""}
            for t in tasks
        }
        outputs: dict[str, str] = {}
        results: dict[str, dict] = {}
        done_ids: set[str] = set()
        failed_ids: set[str] = set()

        def _snapshot() -> list[dict]:
            return [dict(fleet[t["id"]]) for t in tasks]

        # Background flusher pushes the fleet snapshot ~1/s while work runs.
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
            await _flush()  # final frame (all done)

        flusher = asyncio.create_task(_flusher())

        try:
            await self._run_dag(
                tasks, fleet, outputs, results,
                done_ids, failed_ids, signal,
            )
        finally:
            stop.set()
            await flusher

        return self._aggregate(tasks, results)

    async def _run_dag(
        self,
        tasks: list[dict],
        fleet: dict[str, dict],
        outputs: dict[str, str],
        results: dict[str, dict],
        done_ids: set[str],
        failed_ids: set[str],
        signal: asyncio.Event | None,
    ) -> None:
        """Layered topological scheduling: each round runs all ready tasks."""
        while len(done_ids) < len(tasks):
            if signal and signal.is_set():
                break

            # Cascade-skip tasks whose dependencies failed.
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
                        "name": self._label(t), "output": "",
                        "error": "skipped: dependency failed", "steps": 0,
                    }
                    outputs[tid] = ""
                    done_ids.add(tid)
                    failed_ids.add(tid)
                    progressed = True

            # Collect ready tasks (all deps done, none failed).
            ready = [
                t for t in tasks
                if t["id"] not in done_ids
                and all(d in done_ids for d in (t.get("depends_on", []) or []))
            ]

            if not ready:
                if progressed:
                    continue  # skip cascade may have unblocked more
                # No ready, no progress → remaining form a cycle / are stuck.
                for t in tasks:
                    if t["id"] not in done_ids:
                        fleet[t["id"]].update(status="error", action="循环依赖")
                        results[t["id"]] = {
                            "name": self._label(t), "output": "",
                            "error": "unresolved dependency cycle", "steps": 0,
                        }
                        done_ids.add(t["id"])
                        failed_ids.add(t["id"])
                break

            batch = await asyncio.gather(*[
                self._run_task(t, outputs, fleet) for t in ready
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
    ) -> dict:
        """Build the spec (named agent or ad-hoc), inject deps, run it."""
        tid = task["id"]

        def _on_progress(name: str, status: str, step: int, action: str, trace: str = "") -> None:
            fleet[tid].update(status=status, step=step, action=action, trace=trace)

        # Inject dependency outputs into the prompt.
        prompt = task.get("prompt", "")
        dep_blocks = []
        for d in task.get("depends_on", []) or []:
            out = outputs.get(d, "")
            if out:
                dep_blocks.append(f"## 依赖 [{d}] 的产出:\n{out[:_DEP_INJECT_LIMIT]}")
        if dep_blocks:
            prompt = "\n\n".join(dep_blocks) + "\n\n---\n\n" + prompt

        # Resolve spec.
        agent_name = task.get("agent")
        if agent_name:
            named = self._find_named_agent(agent_name)
            if named is None:
                fleet[tid].update(status="error", action="未找到 agent")
                return {"name": agent_name, "output": "",
                        "error": f"named agent '{agent_name}' not found", "steps": 0}
            spec = {
                "name": agent_name,
                "system_prompt": getattr(named, "instructions", TASK_SYSTEM_PROMPT),
                "tools": getattr(named, "agent_tools", []),
                "prompt": prompt,
            }
        else:
            tool_names = task.get("tools") or ["read", "bash"]
            tools = _resolve_tools(tool_names, self._all_tools)
            if not tools:
                fleet[tid].update(status="error", action="无有效工具")
                return {"name": self._label(task), "output": "",
                        "error": f"no valid tools: {tool_names}", "steps": 0}
            spec = {
                "name": self._label(task),
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

    # ── Helpers ─────────────────────────────────────────────

    def _label(self, task: dict) -> str:
        return task.get("agent") or task.get("id") or "task"

    def _find_named_agent(self, name: str) -> AgentTool | None:
        """Find a named agent tool by name (duck-typed: has `instructions`)."""
        for t in self._all_tools:
            if t.name == name and hasattr(t, "instructions"):
                return t
        return None

    def _aggregate(self, tasks: list[dict], results: dict[str, dict]) -> AgentToolResult:
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
            parts.append(f"## [{tid}] {self._label(t)}: {status}\n{body}")
        header = f"Completed {success}/{len(tasks)} tasks:\n\n"
        return AgentToolResult(
            content=[{"type": "text", "text": header + "\n\n---\n\n".join(parts)}],
            details={"results": results, "success": success, "total": len(tasks)},
        )


def _resolve_tools(names: list[str], available: list[AgentTool]) -> list[AgentTool]:
    """Resolve tool names to AgentTool instances."""
    by_name = {t.name: t for t in available}
    return [by_name[n] for n in names if n in by_name]


def create_task_tool(
    model: Model,
    *,
    api_key: str | None = None,
    thinking_level: ThinkingLevel = "off",
    all_tools: list[AgentTool] | None = None,
) -> TaskTool:
    return TaskTool(
        model,
        api_key=api_key,
        thinking_level=thinking_level,
        all_tools=all_tools,
    )
