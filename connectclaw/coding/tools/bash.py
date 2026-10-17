"""Bash tool — execute shell commands with safety checks."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
from typing import Any, Literal

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.coding.safety.sandbox import DirectSandbox, detect_best_sandbox
from connectclaw.coding.safety.shellpath import with_user_path


class BashGuard:
    """Pattern-based dangerous command detection.

    Two tiers:
    - DANGEROUS: always blocked (rm -rf /, mkfs, dd raw, fork bomb, etc.)
    - SUSPICIOUS: require Feishu card authorization (rm, mv, chmod, eval, curl pipe)
    """

    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/(?:\s|$)",    # rm -rf / (root only, followed by space or end)
        r"rm\s+-rf\s+~\s*$",        # rm -rf ~ (home only, followed by space or end)
        r"mkfs\.",                   # format filesystem
        r"\bdd\b[^|]*\bof=/dev/",   # dd to device（of= 在 if= 之前也能命中）
        r">\s*/dev/sd[a-z]",        # redirect to raw device
        r":\(\)\s*\{",              # fork bomb
        r"shutdown\b",              # shutdown
        r"reboot\b",                # reboot
        r"halt\b",                  # halt
        r"poweroff\b",              # poweroff
        r"iptables\b",              # firewall modification
        r"systemctl\s+stop\b",      # stop system services
        r"systemctl\s+disable\b",   # disable system services
    ]

    SUSPICIOUS_PATTERNS = [
        r"\brm\b",                 # any rm
        r"\bmv\b.*/(?:etc|usr|var|opt|bin|sbin|lib|boot)",  # mv to system dir
        r"\bchmod\s+[0-7]*7[0-7]*[0-7]*\b",  # chmod with write/exec
        r"\bchown\b",              # chown
        r"\beval\b",               # eval
        r"\bexec\b",               # exec
        r"\bsource\b",             # source
        r"curl.*\|.*(?:ba)?sh",   # curl pipe to shell
        r"wget.*\|.*(?:ba)?sh",   # wget pipe to shell
        r"\bgit\s+push\s+--force", # force push
        r"\bgit\s+push\s+-f\b",   # force push short
        r"\bdocker\s+rm\b",       # docker remove
        r"\bdocker\s+system\s+prune",  # docker prune
        r"\bnpm\s+publish\b",    # npm publish
        r"\bpip\s+uninstall\b",  # pip uninstall
        r"\bcrontab\s+-r\b",     # remove crontab
        r"\btruncate\b",          # truncate files
    ]

    def __init__(
        self,
        *,
        extra_dangerous: list[str] | None = None,
        extra_suspicious: list[str] | None = None,
    ):
        # Per-instance extension so deployments can tighten the risk profile
        # (e.g. block ``docker system prune`` as DANGEROUS) without forking the
        # class. Never persists to a whitelist — high-risk commands are always
        # gated, the user just decides which ones are in each tier at startup.
        self._extra_dangerous = extra_dangerous or []
        self._extra_suspicious = extra_suspicious or []

    def check(self, command: str) -> Literal["SAFE", "SUSPICIOUS", "DANGEROUS"]:
        for pattern in [*self._extra_dangerous, *self.DANGEROUS_PATTERNS]:
            if re.search(pattern, command, re.IGNORECASE):
                return "DANGEROUS"
        # M16：结构化检查——正则黑名单可被参数换序（rm -fr）、glob（/*）、
        # 子 shell 包裹（(rm -rf /)）、重定向到敏感路径等绕过。
        tier = self._structural_check(command)
        if tier:
            return tier
        for pattern in [*self._extra_suspicious, *self.SUSPICIOUS_PATTERNS]:
            if re.search(pattern, command, re.IGNORECASE):
                return "SUSPICIOUS"
        return "SAFE"

    _COMPOUND_SPLIT_RE = re.compile(r"&&|\|\||[;|&]")
    _BLACKHOLE_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"}

    def _structural_check(self, command: str) -> Literal["SAFE", "SUSPICIOUS", "DANGEROUS"] | None:
        """基于 shlex 分词的补充判定；返回 None 表示无结构性命中。"""
        # 去掉子 shell 括号/命令替换边界，让内层命令可被分词识别
        flat = command.replace("(", " ").replace(")", " ")
        home = os.path.expanduser("~")
        segments = [s.strip() for s in self._COMPOUND_SPLIT_RE.split(flat) if s.strip()]

        # $VAR 有限展开表：跨 segment 收集 VAR=value 赋值（F=/etc/x; echo > $F）
        assignments: dict[str, str] = {}
        for segment in segments:
            try:
                tokens = shlex.split(segment)
            except ValueError:
                tokens = segment.split()
            if tokens and re.fullmatch(r"[A-Za-z_]\w*", tokens[0].split("=", 1)[0]) \
                    and "=" in tokens[0] and not tokens[0].startswith("="):
                var, val = tokens[0].split("=", 1)
                assignments[var] = val.strip("\"'")

        def _expand(target: str) -> str:
            if target.startswith("$"):
                var = target[1:].lstrip("{").rstrip("}")
                if var in assignments:
                    return assignments[var]
            return os.path.expanduser(target)

        for segment in segments:
            try:
                tokens = shlex.split(segment)
            except ValueError:
                tokens = segment.split()
            if not tokens:
                continue
            name = os.path.basename(tokens[0])

            if name == "rm":
                targets = [t for t in tokens[1:] if not t.startswith("-")]
                for tgt in targets:
                    if tgt in ("/*", "$HOME", "${HOME}"):
                        return "DANGEROUS"
                    expanded = os.path.expanduser(tgt).rstrip("/")
                    if expanded in ("", "/", home):
                        return "DANGEROUS"
            elif name == "chmod" and "-R" in tokens[1:]:
                targets = [t for t in tokens[2:] if not t.startswith("-")]
                if any(os.path.expanduser(t).rstrip("/") in ("", "/") for t in targets):
                    return "DANGEROUS"

            # 写路径目标收集：重定向（含引号包裹，shlex 分词已去引号）+ tee
            # Review 2026-10-17：原正则字符类排除引号，`> "/etc/passwd"` 直接绕过。
            targets: list[str] = []
            for m in re.finditer(r">>?>?\s*(\"[^\"]*\"|'[^']*'|[^\s;|&]+)", segment):
                targets.append(m.group(1).strip("\"'"))
            if name == "tee":
                targets.extend(t for t in tokens[1:] if not t.startswith("-"))
            for target in targets:
                target = _expand(target)
                if target in self._BLACKHOLE_TARGETS:
                    continue
                if re.match(r"/dev/(?:sd|nvme|mapper)", target):
                    return "DANGEROUS"
                if target.startswith(("/etc/", "/boot/", "/proc/", "/sys/")) or "/.ssh/" in target:
                    return "DANGEROUS"
        return None


class BashTool(AgentTool):
    name = "bash"
    label = "bash"
    description = (
        "Execute a shell command in a subprocess. "
        "The command will be executed in the working directory. "
        "A timeout can be specified (default 120 seconds). "
        "Runs in a sandbox (filesystem isolation) with full network access; "
        "dangerous commands require user authorization. "
        "For long-lived processes use background=true and manage them with the "
        "bg_tasks tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 120, max: 600)",
            },
            "background": {
                "type": "boolean",
                "description": (
                    "Set true to launch as a detached long-lived task "
                    "(dev servers, watchers, trace viewers) instead of "
                    "waiting. Returns a task id immediately; the process "
                    "keeps running across bash calls and its output goes "
                    "to a log file. Manage with the bg_tasks tool "
                    "(list/tail/kill). Tasks stop when the bot restarts. "
                    "timeout is ignored in background mode."
                ),
            },
        },
        "required": ["command"],
    }

    def __init__(self, cwd: str, guard: BashGuard | None = None,
                 sandbox_cls: type | None = None):
        self._cwd = cwd
        self._guard = guard or BashGuard()
        # sandbox_cls is the operator's [sandbox] mode resolved to a class
        # (see sandbox.resolve_sandbox_cls); None keeps runtime auto-detection.
        self._sandbox_cls = sandbox_cls or detect_best_sandbox()
        # Auto-detect git root for read-write bind in sandbox
        self._git_root = self._find_git_root()

    def _find_git_root(self) -> str | None:
        """Walk up from cwd to find the nearest git repository root."""
        try:
            result = subprocess.run(
                ["git", "-C", self._cwd, "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                root = result.stdout.strip()
                if root and root != self._cwd and root.startswith('/'):
                    return root
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass
        return None

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
        unsandboxed: bool = False,
    ) -> AgentToolResult:
        command = params["command"]
        timeout = min(params.get("timeout", 120) or 120, 600)

        # Safety check
        check = self._guard.check(command)
        if check == "DANGEROUS":
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": (
                        f"命令因安全原因被阻止：`{command}`\n"
                        f"该命令匹配危险模式，无法执行。"
                    ),
                }],
            )

        # Run with the user's interactive PATH (the systemd env PATH misses
        # ~/.npm-global/bin, ~/.local/bin, pyenv, ...).
        command = with_user_path(command)

        if unsandboxed:
            # Operator escape channel (`!`/`/bash`): identity-gated HUMAN
            # commands run on the real filesystem — the sandbox would block
            # the very maintenance work (e.g. editing config.toml) the escape
            # exists for. Risk control here is the operator open_id whitelist
            # + BashGuard (DANGEROUS hard block, SUSPICIOUS auth card).
            # The agent's own bash tool NEVER gets unsandboxed=True: its
            # isolation (incl. the RO bind on the escape-hatch configs) is
            # the agent's privilege boundary.
            sandbox = DirectSandbox(cwd=self._cwd)
        else:
            # Build allowed paths: cwd + git root (for git operations)
            allowed_paths = [self._cwd]
            if self._git_root:
                allowed_paths.append(self._git_root)

            sandbox = self._sandbox_cls(
                cwd=self._cwd,
                allowed_paths=allowed_paths,
            )

        # Background mode: the bot process spawns the *whole* sandbox command
        # line (bwrap argv included) and supervises it, so the task outlives
        # this call. BashGuard above is NOT bypassed here.
        if params.get("background"):
            from ..bg_tasks import get_bg_manager
            try:
                task = await get_bg_manager().start(
                    params["command"], sandbox.build(command), self._cwd)
            except Exception as e:
                return AgentToolResult(
                    content=[{"type": "text",
                              "text": f"后台任务启动失败：{e}"}],
                    details={"is_error": True},
                )
            return AgentToolResult(
                content=[{"type": "text", "text": (
                    f"[background] 任务 {task.task_id} 已启动"
                    f"（pid {task.pid}, {sandbox.level.value}）。\n"
                    f"日志：{task.log_path}\n"
                    f"它跨 bash 调用持续运行，bot 重启即终止。"
                    f"读输出用 bg_tasks(action=tail, id={task.task_id})，"
                    f"停止用 bg_tasks(action=kill, id={task.task_id})。"
                )}],
                details={"task_id": task.task_id, "pid": task.pid,
                         "background": True},
            )

        result = await sandbox.execute(command, timeout=timeout)

        # ── Build output visible to LLM ────────────────────────
        # Codex JsonToolOutput 同款：结构化 JSON 作为 tool 消息内容——
        # exit_code / wall_time 是固定字段，模型不再需要从文本里猜。
        payload: dict[str, Any] = {
            "output": result.stdout.rstrip(),
            "exit_code": result.exit_code,
            "wall_time_s": round(result.wall_time_ms / 1000, 1),
        }
        if result.stderr.strip():
            payload["stderr"] = result.stderr.rstrip()
        if result.timed_out:
            payload["timed_out"] = True
            payload["timeout_s"] = timeout
        if result.truncated:
            payload["truncated"] = True
        output = json.dumps(payload, ensure_ascii=False)

        details = {
            "exit_code": result.exit_code,
            "command": command,
            "sandbox_level": result.level.value,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
        }

        return AgentToolResult(
            content=[{"type": "text", "text": output}],
            details=details,
        )


def create_bash_tool(cwd: str, guard: BashGuard | None = None,
                    sandbox_cls: type | None = None) -> BashTool:
    return BashTool(cwd, guard, sandbox_cls)
