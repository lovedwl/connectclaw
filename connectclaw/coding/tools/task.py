"""
Task tool — spawn parallel sub-agents for independent tasks.

DAG model: all sub-agents run concurrently, results aggregated when all complete.
Each sub-agent gets a restricted tool set and runs a full agent loop.

Example usage from main agent:
  task(tasks=[
    {"prompt": "Check Python type errors", "tools": ["read", "bash"]},
    {"prompt": "Run the test suite", "tools": ["read", "bash"]},
  ])
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from connectclaw.agent.agent import Agent
from connectclaw.agent.types import AgentTool, AgentToolResult, ThinkingLevel
from connectclaw.provider.types import Model


TASK_SYSTEM_PROMPT = """You are a focused sub-agent. Complete the assigned task and return results.

## Rules
- Only use the tools you've been given
- Be concise — return results, not conversation
- If you can't complete the task, explain why
- Use the bash tool for commands, read for files
- When done, output a clear summary of findings
"""


class TaskTool(AgentTool):
    """Spawn parallel sub-agents. All run concurrently, results aggregated."""

    name = "task"
    label = "task"
    description = (
        "Spawn one or more sub-agents to work on independent tasks in parallel. "
        "Each sub-agent gets a restricted set of tools. "
        "All sub-agents run concurrently and results are returned when all complete. "
        "Use this to parallelize independent work like checking types + running tests."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "List of tasks to run in parallel",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The task for the sub-agent to complete",
                        },
                        "tools": {
                            "type": "array",
                            "description": "Tool names the sub-agent can use (default: ['read', 'bash'])",
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

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        tasks = params["tasks"]
        if not tasks:
            return AgentToolResult(
                content=[{"type": "text", "text": "No tasks provided."}],
            )

        # Spawn all sub-agents in parallel
        results = await asyncio.gather(*[
            self._run_sub_agent(i, task, signal)
            for i, task in enumerate(tasks)
        ])

        # Aggregate
        output_parts = []
        success_count = 0
        for i, (task, result) in enumerate(zip(tasks, results)):
            status = "OK" if not result.get("error") else "ERROR"
            if not result.get("error"):
                success_count += 1
            output_parts.append(
                f"## Task {i+1}: {status}\n"
                f"**Prompt:** {task['prompt'][:200]}\n\n"
                f"{result.get('output', '(no output)')}\n"
            )

        return AgentToolResult(
            content=[{
                "type": "text",
                "text": f"Completed {success_count}/{len(tasks)} tasks:\n\n" + "\n---\n".join(output_parts),
            }],
            details={
                "results": results,
                "success": success_count,
                "total": len(tasks),
            },
        )

    async def _run_sub_agent(
        self,
        idx: int,
        task: dict,
        signal: asyncio.Event | None,
    ) -> dict:
        """Run a single sub-agent and return its result."""
        tool_names = task.get("tools", ["read", "bash"])
        tools = _resolve_tools(tool_names, self._all_tools)

        if not tools:
            return {
                "task_index": idx,
                "error": f"No valid tools found: {tool_names}",
                "output": "",
            }

        sub_agent = Agent(
            system_prompt=TASK_SYSTEM_PROMPT,
            model=self._model,
            thinking_level=self._thinking_level,
            tools=tools,
        )

        try:
            await sub_agent.prompt(task["prompt"])

            # Extract final result from the last assistant message
            output = _extract_assistant_text(sub_agent.state.messages)
            return {
                "task_index": idx,
                "output": output,
                "tool_count": len(tools),
                "message_count": len(sub_agent.state.messages),
            }
        except Exception as e:
            return {
                "task_index": idx,
                "error": str(e),
                "output": "",
            }


def _resolve_tools(
    names: list[str],
    available: list[AgentTool],
) -> list[AgentTool]:
    """Resolve tool names to AgentTool instances."""
    by_name = {t.name: t for t in available}
    return [by_name[n] for n in names if n in by_name]


def _extract_assistant_text(messages: list[Any]) -> str:
    """Extract text from the last assistant message."""
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        content = msg.content
        if isinstance(content, str):
            return content
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    return "(no assistant response)"


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
