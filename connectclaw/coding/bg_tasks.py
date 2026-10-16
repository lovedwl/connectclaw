"""Background tasks — long-lived processes that outlive a single bash call.

Why this exists: every foreground bash call runs inside a fresh
``bwrap --unshare-pid`` whose PID namespace is torn down the moment the call
returns, so any daemon started from it is reaped with the call. Background
mode flips who owns the process: the (long-lived) bot process spawns and
supervises the sandbox directly, so servers / watchers / dev loops keep
running across tool calls.

Lifetime contract: tasks live until the bot exits (the sandboxed ones via
``--die-with-parent``). Things that must survive a bot restart belong under
systemd on the host, not here.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import glob
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult


@dataclass
class BgTask:
    task_id: str
    command: str          # display command (no PATH wrapper)
    argv0: str            # first argv element, e.g. "bwrap" or "bash"
    pid: int
    started_at: float
    log_path: str
    proc: asyncio.subprocess.Process = field(repr=False)
    exit_code: int | None = None
    killed: bool = False

    @property
    def running(self) -> bool:
        return self.proc.returncode is None

    @property
    def state(self) -> str:
        if self.running:
            return "running"
        return "killed" if self.killed else "exited"

    @property
    def uptime(self) -> int:
        return int(time.time() - self.started_at)


class BgTaskManager:
    """Spawn / supervise / kill background tasks owned by the bot process."""

    MAX_LOGS = 50  # keep at most this many finished-task logs on disk
    MAX_FINISHED = 20  # M25：内存里最多保留这么多个已结束任务（_tasks 只增不减的解药）

    def __init__(self, log_dir: str | None = None):
        self.log_dir = log_dir or os.path.expanduser("~/.connectclaw/tasks")
        self._tasks: dict[str, BgTask] = {}
        self._seq = 0
        self._watchers: set[asyncio.Task] = set()
        # M25：bot 退出时杀掉所有存活后台任务——mode=none/DirectSandbox 的
        # 任务没有 --die-with-parent，不杀就成孤儿（与文档"存活到 bot 退出"
        # 的契约一致：退出即回收）。
        atexit.register(self._kill_all_sync)

    async def start(self, command: str, argv: list[str], cwd: str) -> BgTask:
        os.makedirs(self.log_dir, exist_ok=True)
        self._seq += 1
        task_id = f"t{self._seq:03d}"
        log_path = os.path.join(self.log_dir, f"{task_id}.log")
        with open(log_path, "ab") as fh:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=fh,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        task = BgTask(
            task_id=task_id, command=command, argv0=os.path.basename(argv[0]),
            pid=proc.pid, started_at=time.time(), log_path=log_path, proc=proc,
        )
        self._tasks[task_id] = task
        watcher = asyncio.create_task(self._reap(task))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        self._prune_logs()
        return task

    async def _reap(self, task: BgTask) -> None:
        with contextlib.suppress(Exception):
            task.exit_code = await task.proc.wait()
        # M25：已结束任务只保留最近 MAX_FINISHED 条，防止 _tasks 缓慢增长
        finished = [t for t in self._tasks.values() if not t.running]
        for t in sorted(finished, key=lambda t: t.started_at)[:-self.MAX_FINISHED]:
            self._tasks.pop(t.task_id, None)

    def _kill_all_sync(self) -> None:
        """atexit：同步杀掉所有存活后台任务（事件循环已停，只能发信号）。"""
        for t in list(self._tasks.values()):
            if t.running:
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(os.getpgid(t.pid), signal.SIGKILL)

    def list_tasks(self) -> list[BgTask]:
        return list(self._tasks.values())

    def get(self, task_id: str) -> BgTask | None:
        return self._tasks.get(task_id)

    async def kill(self, task_id: str) -> tuple[bool, str]:
        task = self._tasks.get(task_id)
        if task is None:
            return False, f"未找到任务 {task_id}"
        if not task.running:
            return False, f"任务 {task_id} 已经结束（{task.state}）"
        task.killed = True  # set before signalling so _reap classifies it
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(task.pid), signal.SIGTERM)
        try:
            await asyncio.wait_for(task.proc.wait(), timeout=3)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(task.pid), signal.SIGKILL)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task.proc.wait(), timeout=3)
        return True, f"任务 {task_id} 已终止"

    def tail(self, task_id: str, max_bytes: int | str | None = 4000) -> tuple[bool, str]:
        task = self._tasks.get(task_id)
        if task is None:
            return False, f"未找到任务 {task_id}"
        # M25：max_bytes 非数字入参不再抛 ValueError 炸掉工具调用
        try:
            max_bytes = int(max_bytes)
        except (TypeError, ValueError):
            max_bytes = 4000
        max_bytes = max(1, min(max_bytes, 20000))
        try:
            with open(task.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                data = f.read()
        except OSError as e:
            return False, f"读取日志失败：{e}"
        text = data.decode("utf-8", errors="replace")
        if size > len(data):
            text = f"…（前面 {size - len(data)} 字节已省略）…\n" + text
        return True, text

    def _prune_logs(self) -> None:
        """Keep only the newest MAX_LOGS finished-task logs (never a live one)."""
        try:
            files = sorted(glob.glob(os.path.join(self.log_dir, "*.log")),
                           key=os.path.getmtime)
        except OSError:
            return
        running = {t.log_path for t in self._tasks.values() if t.running}
        stale = [f for f in files if f not in running]
        for f in (stale[:-self.MAX_LOGS] if self.MAX_LOGS else stale):
            with contextlib.suppress(OSError):
                os.remove(f)


_MANAGER: BgTaskManager | None = None


def get_bg_manager() -> BgTaskManager:
    """Process-wide singleton (tests replace the module-level _MANAGER)."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = BgTaskManager()
    return _MANAGER


class BgTasksTool(AgentTool):
    """Control plane for tasks launched with bash(background=true)."""

    name = "bg_tasks"
    label = "bg_tasks"
    description = (
        "Inspect and control background processes started with "
        "bash(background=true): list them, tail their combined "
        "stdout/stderr log, or kill them. Background tasks survive across "
        "bash calls but stop when the bot restarts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "tail", "kill"],
                "description": "list: overview of all tasks; tail: read the "
                               "end of a task's log; kill: terminate a task.",
            },
            "id": {
                "type": "string",
                "description": "Task id (e.g. t001). Required for tail/kill.",
            },
            "max_bytes": {
                "type": "integer",
                "description": "tail only: max trailing bytes to read "
                               "(default 4000, max 20000).",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
        unsandboxed: bool = False,
    ) -> AgentToolResult:
        mgr = get_bg_manager()
        action = params.get("action", "list")
        is_error = False

        if action == "list":
            tasks = mgr.list_tasks()
            if not tasks:
                text = "没有后台任务。用 bash 的 background=true 参数启动一个。"
            else:
                rows = []
                for t in tasks:
                    extra = (f"up={self._fmt_dur(t.uptime)}" if t.running
                             else f"exit={t.exit_code}")
                    rows.append(
                        f"{t.task_id}  {t.state:<7} pid={t.pid} [{t.argv0}] {extra}\n"
                        f"     cmd: {t.command}\n"
                        f"     log: {t.log_path}"
                    )
                text = "\n".join(rows)
        elif action == "tail":
            task_id = (params.get("id") or "").strip()
            if not task_id:
                text, is_error = "tail 需要 id 参数", True
            else:
                # M25：非法 max_bytes 不再炸 ValueError
                try:
                    max_bytes = min(int(params.get("max_bytes") or 4000), 20000)
                except (TypeError, ValueError):
                    max_bytes = 4000
                ok, text = mgr.tail(task_id, max_bytes)
                is_error = not ok
                if ok:
                    t = mgr.get(task_id)
                    text = (f"[{task_id} {t.state} exit={t.exit_code}]\n" + text
                            if text.strip() else f"[{task_id}] 日志为空")
        elif action == "kill":
            task_id = (params.get("id") or "").strip()
            if not task_id:
                text, is_error = "kill 需要 id 参数", True
            else:
                ok, text = await mgr.kill(task_id)
                is_error = not ok
        else:
            text, is_error = f"未知 action：{action}", True

        details: dict[str, Any] = {"action": action}
        if is_error:
            details["is_error"] = True
        return AgentToolResult(
            content=[{"type": "text", "text": text}], details=details,
        )

    @staticmethod
    def _fmt_dur(seconds: int) -> str:
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m{seconds % 60:02d}s"
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
