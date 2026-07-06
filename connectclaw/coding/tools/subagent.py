"""
Shared sub-agent runner — spawn a child Agent and bubble its progress.

Used by:
  - the task tool (parallel / DAG orchestration of many sub-agents)
  - named agents (a `.md`-defined agent invoked as a tool)

Progress is surfaced through an `on_progress` callback so the caller (e.g. the
task tool's fleet card) can render a live view.

Design note: the child agent's own live-card callbacks are deliberately NOT
wired to the parent's Feishu card. Instead we subscribe to its coarse event
stream and report a compact per-agent status. That way N sub-agents collapse
into one fleet-card row each, rather than flooding the parent card with every
sub-tool-call — which is exactly the "information explosion" we want to avoid.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from connectclaw.agent.agent import Agent
from connectclaw.agent.harness.messages import convert_to_llm
from connectclaw.agent.types import AgentTool, ThinkingLevel
from connectclaw.provider.types import Model

# on_progress(name, status, step, action, trace)
#   status ∈ {"waiting", "running", "done", "error"}
#   trace  = markdown timeline of the child's 💭 thinking + 🔧 tool calls
ProgressCallback = Callable[[str, str, int, str, str], None]


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def _summarize_args(args: dict) -> str:
    """Pick the most telling arg (path/command/query/…) as a short inline code."""
    if not isinstance(args, dict) or not args:
        return ""
    for k in ("command", "path", "file_path", "query", "url", "pattern", "prompt", "name"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return f"`{_clip(v, 60)}`"
    for v in args.values():
        if isinstance(v, str) and v.strip():
            return f"`{_clip(v, 60)}`"
    return ""


def _result_text(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    for b in result.get("content") or []:
        if isinstance(b, dict) and b.get("type") == "text":
            return _clip(b.get("text") or "", 120)
    return ""


class _TraceBuilder:
    """Fold a child agent's event stream into a compact markdown trace.

    Interleaves 💭 thinking (from assistant message_end) with 🔧 tool-call →
    result lines (tool args captured at start, result at end, joined by id so
    concurrent child tools never cross wires). Ordering follows event arrival,
    which is already chronological in the loop.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._pending: dict[str, tuple[str, str]] = {}

    def feed(self, event: dict) -> None:
        etype = event.get("type", "")
        if etype == "message_end":
            msg = event.get("message")
            if getattr(msg, "role", None) == "assistant":
                for b in getattr(msg, "content", None) or []:
                    if isinstance(b, dict) and b.get("type") == "thinking":
                        th = (b.get("thinking") or "").strip()
                        if th:
                            self.lines.append(f"💭 {_clip(th, 200)}")
        elif etype == "tool_execution_start":
            tid = event.get("tool_call_id", "")
            self._pending[tid] = (
                event.get("tool_name") or "工具",
                _summarize_args(event.get("args") or {}),
            )
        elif etype == "tool_execution_end":
            tid = event.get("tool_call_id", "")
            name, argsum = self._pending.pop(tid, (event.get("tool_name") or "工具", ""))
            line = f"🔧 {name}"
            if argsum:
                line += f" {argsum}"
            rtext = _result_text(event.get("result") or {})
            if rtext:
                line += f" → {rtext}"
            self.lines.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def extract_assistant_text(messages: list[Any]) -> str:
    """Extract text from the last assistant message."""
    for msg in reversed(messages):
        if getattr(msg, "role", None) != "assistant":
            continue
        content = msg.content
        if isinstance(content, str):
            return content
        texts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts)
    return "(no assistant response)"


async def run_subagent(
    spec: dict[str, Any],
    *,
    model: Model,
    api_key: str | None = None,
    thinking_level: ThinkingLevel = "off",
    on_progress: ProgressCallback | None = None,
    signal: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Spawn a single child agent from `spec` and return its result.

    spec = {
      "name": str,            # display name (fleet row label)
      "system_prompt": str,   # child agent's system prompt
      "tools": list[AgentTool],
      "prompt": str,          # the task to run
    }

    Returns {"name", "output", "error", "steps"}.
    Never raises — errors are captured into the returned dict.
    """
    name = spec.get("name", "agent")
    tools: list[AgentTool] = spec.get("tools") or []
    system_prompt = spec.get("system_prompt") or ""
    prompt = spec.get("prompt", "")

    counter = {"step": 0, "action": "启动"}
    trace = _TraceBuilder()

    def _report(status: str) -> None:
        if on_progress:
            on_progress(name, status, counter["step"], counter["action"], trace.text)

    def _listener(event: dict) -> None:
        etype = event.get("type", "")
        trace.feed(event)
        if etype == "tool_execution_start":
            counter["action"] = event.get("tool_name", "") or "工具"
            _report("running")
        elif etype == "tool_execution_end":
            counter["step"] += 1
            _report("running")
        elif etype == "message_end":
            # Assistant turn done (thinking now captured) — refresh so 💭 shows
            # before the tools it decided to call.
            if getattr(event.get("message"), "role", None) == "assistant":
                _report("running")

    agent = Agent(
        system_prompt=system_prompt,
        model=model,
        thinking_level=thinking_level,
        tools=tools,
        convert_to_llm=convert_to_llm,
        get_api_key=lambda _: api_key,
    )
    agent.subscribe(_listener)

    # Bridge the parent's abort signal to the child agent.
    abort_watcher: asyncio.Task | None = None
    if signal is not None:
        async def _watch() -> None:
            await signal.wait()
            agent.abort()
        abort_watcher = asyncio.create_task(_watch())

    _report("running")
    try:
        await agent.prompt(prompt)
        output = extract_assistant_text(agent.state.messages)
        err = agent.state.error_message
        if err:
            counter["action"] = "出错"
            _report("error")
            return {"name": name, "output": output, "error": err, "steps": counter["step"]}
        counter["action"] = "完成"
        _report("done")
        return {"name": name, "output": output, "error": None, "steps": counter["step"]}
    except Exception as e:  # noqa: BLE001
        counter["action"] = "出错"
        _report("error")
        return {"name": name, "output": "", "error": str(e), "steps": counter["step"]}
    finally:
        if abort_watcher is not None:
            abort_watcher.cancel()
