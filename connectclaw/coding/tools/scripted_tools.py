"""
Scripted tools — "a tool is a frontmatter script", same shape as named agents.

A `*.tool.md` file:
    ---
    name: browser
    desc: 操控无头浏览器(Lightpanda),会话内保持登录态
    runtime: python          # python | bash | sh | node
    stateful: true           # long-lived serve-loop process, per conversation
    params: { op: "goto|read|click|type|search", arg: "…" }
    ---
    <body: script source>

I/O contract (uniform for both modes): the body reads ONE line of JSON (the
params) from stdin and writes its result to stdout.
  - stateless (default): harness spawns the interpreter per call, feeds one line,
    reads stdout, waits for exit. The body handles one line then exits.
  - stateful: harness keeps ONE process alive and feeds a line per call; the body
    loops (`for line in sys.stdin: … print(result, flush=True)`). Kept per
    conversation (login state never crosses chats), idle-reaped, crash-restarted.

Why not MCP/plugins: this is just a file + (optionally) a long-lived subprocess
fed text lines — no protocol stack, no in-process ABI, language-agnostic, crash
isolated by the process boundary. `.tool.json` (dynamic.py) stays as L0.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.coding.tools.dynamic import load_dynamic_tools
from connectclaw.coding.tools.named_agents import _NAME_RE, _split_frontmatter
from connectclaw.coding.tools.tool_context import current_conversation, current_subagent

# current_conversation / current_subagent are defined in tool_context (imported
# above, dependency-free to avoid import cycles) and re-exported here for
# existing callers (e.g. coding_agent imports current_conversation from here).

# runtime name → interpreter argv that takes the body as its next arg
_RUNTIMES: dict[str, list[str]] = {
    "python": [sys.executable, "-c"],
    "bash": ["bash", "-c"],
    "sh": ["sh", "-c"],
    "node": ["node", "-e"],
}

_DEFAULT_CALL_TIMEOUT = 120
_DEFAULT_IDLE_TIMEOUT = 300


# ── Parsing ─────────────────────────────────────────────────────


def _params_schema(params: dict | None) -> dict[str, Any]:
    """Turn the minimal `{name: "desc"}` frontmatter into a JSON Schema.

    All params are string and optional (the script validates what it needs) —
    keeps authoring cheap; agents rarely need typed/required here.
    """
    props: dict[str, Any] = {}
    for k, v in (params or {}).items():
        props[str(k)] = {"type": "string", "description": str(v)}
    return {"type": "object", "properties": props, "required": []}


def parse_tool_file(path: str | Path) -> dict[str, Any] | None:
    """Parse a `*.tool.md`. Returns {name,desc,runtime,stateful,parameters,body} or None."""
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

    name = str(meta.get("name") or Path(path).stem.replace(".tool", ""))
    if not _NAME_RE.match(name):
        return None
    runtime = str(meta.get("runtime") or "bash").lower()
    return {
        "name": name,
        "desc": str(meta.get("desc") or meta.get("description") or f"Scripted tool: {name}"),
        "runtime": runtime,
        "stateful": bool(meta.get("stateful", False)),
        "parameters": _params_schema(meta.get("params") if isinstance(meta.get("params"), dict) else {}),
        "body": body.strip(),
    }


# ── Stateless execution ─────────────────────────────────────────


async def _run_once(cmd: list[str], body: str, params: dict, timeout: int) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd, body,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(json.dumps(params).encode() + b"\n"), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"[tool timed out after {timeout}s]"
    text = out.decode("utf-8", "replace").strip()
    if proc.returncode:
        etext = err.decode("utf-8", "replace").strip()
        text = (text + f"\n[exit {proc.returncode}] {etext}").strip()
    return text


# ── Stateful session runtime ────────────────────────────────────


class _Session:
    def __init__(self, proc: asyncio.subprocess.Process):
        self.proc = proc
        self.lock = asyncio.Lock()
        self.last = time.monotonic()


class SessionRuntime:
    """Keeps one long-lived process per (conversation, tool), fed one JSON line
    per call. Idle-reaped, crash-restarted, isolated per conversation."""

    def __init__(self, idle_timeout: int = _DEFAULT_IDLE_TIMEOUT):
        self._sessions: dict[tuple[str, str, str], _Session] = {}
        self._spawn_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self._idle = idle_timeout
        self._reaper: asyncio.Task | None = None

    def _key(self, tool: str) -> tuple[str, str, str]:
        # (conversation, subagent, tool): each fleet member gets its OWN session,
        # so N sub-agents run N browsers in parallel instead of contending on one.
        return (current_conversation.get(), current_subagent.get(), tool)

    def _spawn_lock(self, key: tuple[str, str, str]) -> asyncio.Lock:
        lock = self._spawn_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._spawn_locks[key] = lock
        return lock

    async def call(self, tool: str, runtime: str, body: str, params: dict,
                   timeout: int = _DEFAULT_CALL_TIMEOUT) -> str:
        cmd = _RUNTIMES.get(runtime)
        if not cmd:
            return f"[unknown runtime '{runtime}']"
        key = self._key(tool)
        last_err = "unknown"
        for _ in (1, 2):  # restart once on a dead/crashed session
            sess = self._sessions.get(key)
            if sess is None or sess.proc.returncode is not None:
                # Serialize first-spawn so concurrent callers don't each start a
                # process (which would leak all but the last). Double-check inside.
                async with self._spawn_lock(key):
                    sess = self._sessions.get(key)
                    if sess is None or sess.proc.returncode is not None:
                        sess = await self._spawn(key, cmd, body)
            async with sess.lock:
                try:
                    assert sess.proc.stdin and sess.proc.stdout
                    sess.proc.stdin.write(json.dumps(params).encode() + b"\n")
                    await sess.proc.stdin.drain()
                    line = await asyncio.wait_for(sess.proc.stdout.readline(), timeout=timeout)
                    if not line:
                        raise RuntimeError("session process closed the pipe")
                    sess.last = time.monotonic()
                    return line.decode("utf-8", "replace").strip()
                except Exception as e:  # noqa: BLE001
                    last_err = str(e)
                    await self._kill(key)
        return f"[stateful tool '{tool}' failed: {last_err}]"

    async def _spawn(self, key: tuple[str, str, str], cmd: list[str], body: str) -> _Session:
        proc = await asyncio.create_subprocess_exec(
            *cmd, body,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        sess = _Session(proc)
        self._sessions[key] = sess
        self._ensure_reaper()
        return sess

    async def _kill(self, key: tuple[str, str, str]) -> None:
        sess = self._sessions.pop(key, None)
        if sess is not None:
            try:
                sess.proc.kill()
            except Exception:
                pass

    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap())

    async def _reap(self) -> None:
        while self._sessions:
            await asyncio.sleep(min(self._idle, 30))
            now = time.monotonic()
            for key, sess in list(self._sessions.items()):
                if now - sess.last > self._idle or sess.proc.returncode is not None:
                    await self._kill(key)

    async def shutdown(self, conversation_key: str) -> None:
        """Kill all sessions for one conversation (e.g. on /new or close)."""
        for key in [k for k in self._sessions if k[0] == conversation_key]:
            await self._kill(key)

    async def shutdown_all(self) -> None:
        for key in list(self._sessions):
            await self._kill(key)


# ── ScriptedTool ────────────────────────────────────────────────


def _unwrap_stateful(raw: str) -> str:
    """A stateful body answers ONE line of JSON `{ok, result|error}` (markdown
    is multi-line, so it can't ride the line protocol raw). Unwrap to the text
    the agent should see; plain (non-JSON) lines pass through unchanged."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "ok" in obj:
            return obj.get("result", "") if obj.get("ok") else f"[error] {obj.get('error', '')}"
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


class ScriptedTool(AgentTool):
    """A frontmatter-script tool. Stateless → run body per call; stateful →
    delegate to a per-conversation SessionRuntime process."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict,
        runtime: str,
        body: str,
        stateful: bool,
        session_runtime: SessionRuntime,
    ):
        self.name = name
        self.label = name
        self.description = description
        self.parameters = parameters
        self._runtime = runtime
        self._body = body
        self._stateful = stateful
        self._rt = session_runtime

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        cmd = _RUNTIMES.get(self._runtime)
        if not cmd:
            return AgentToolResult(content=[{
                "type": "text",
                "text": f"Tool '{self.name}': unknown runtime '{self._runtime}'.",
            }])
        if self._stateful:
            text = _unwrap_stateful(
                await self._rt.call(self.name, self._runtime, self._body, params)
            )
        else:
            text = await _run_once(cmd, self._body, params, _DEFAULT_CALL_TIMEOUT)
        return AgentToolResult(content=[{"type": "text", "text": text or "(no output)"}])


# ── Loader ──────────────────────────────────────────────────────


def load_scripted_tools(
    dirs: list[str],
    session_runtime: SessionRuntime,
    cwd: str,
) -> list[AgentTool]:
    """Scan dirs for `*.tool.md` (frontmatter scripts) + `*.tool.json` (L0).

    Earlier dirs win on name conflicts (builtin before user). Safe every turn.
    """
    out: list[AgentTool] = []
    seen: set[str] = set()
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*.tool.md")):
            meta = parse_tool_file(f)
            if not meta or meta["name"] in seen:
                continue
            seen.add(meta["name"])
            out.append(ScriptedTool(
                name=meta["name"],
                description=meta["desc"],
                parameters=meta["parameters"],
                runtime=meta["runtime"],
                body=meta["body"],
                stateful=meta["stateful"],
                session_runtime=session_runtime,
            ))
        for t in load_dynamic_tools(str(p), cwd):  # L0 .tool.json
            if t.name not in seen:
                seen.add(t.name)
                out.append(t)
    return out
