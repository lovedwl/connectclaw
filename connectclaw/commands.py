"""
Slash commands for ConnectClaw.

A command is matched against the stripped message text. It may be a bare
"/name" (exact match) or "/name <args>" — the leading token selects the
handler and the remainder is passed to it as `args`.
Add new commands with the @register decorator below.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from connectclaw.logging import get_logger

logger = get_logger(__name__)


# ── Public API ─────────────────────────────────────────────────


async def handle(
    text: str,
    *,
    conversation_key: str,
    agent: Any,  # CodingAgent
) -> str | None:
    """Try to handle a slash command. Returns response string or None (not a command).
    Unknown slash commands get a help listing — they never fall through to the agent."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    # Exact match first (bare commands); else split "/cmd rest" into name + args.
    cmd = COMMANDS.get(stripped)
    args = ""
    if cmd is None:
        head, _, rest = stripped.partition(" ")
        cmd = COMMANDS.get(head)
        args = rest.strip()
    if cmd is None:
        return _help_text(stripped)

    # A command failure must surface to the user, not vanish into logs — the
    # channel's outer handler only logs exceptions, it doesn't reply.
    try:
        return await cmd.handler(conversation_key, agent, args)
    except Exception as e:
        logger.error("Command %s failed: %s", cmd.name, e)
        return f"命令 {cmd.name} 执行出错：{e}"


def _help_text(unknown: str = "") -> str:
    """Build a help listing for available commands."""
    lines = []
    if unknown:
        lines.append(f"Unknown command: {unknown}\n")
    lines.append("**Available commands:**")
    for name, cmd in sorted(COMMANDS.items()):
        lines.append(f"- **{name}** — {cmd.description}")
    return "\n".join(lines)


def register(name: str, description: str) -> Callable:
    """Decorator to register a command handler.

    Usage:
        @register("/foo", "does something")
        async def _foo(conversation_key, agent, args) -> str:
            return "done"
    """
    def decorator(fn: CommandFn) -> CommandFn:
        COMMANDS[name] = Command(name=name, description=description, handler=fn)
        return fn
    return decorator


# ── Internals ──────────────────────────────────────────────────


CommandFn = Callable[[str, Any, str], Awaitable[str]]


class Command:
    __slots__ = ("name", "description", "handler")

    def __init__(self, *, name: str, description: str, handler: CommandFn):
        self.name = name
        self.description = description
        self.handler = handler


COMMANDS: dict[str, Command] = {}


# ── Built-in Commands ──────────────────────────────────────────


@register("/stop", "interrupt the running agent loop")
async def _stop(conversation_key: str, agent: Any, args: str = "") -> str:
    agent.abort(conversation_key)
    return "⏹ Interrupted."


@register("/new", "start a fresh conversation (clear context)")
async def _new(conversation_key: str, agent: Any, args: str = "") -> str:
    await agent.new_session(conversation_key)
    return "🆕 Fresh conversation started."


@register(
    "/memory",
    "查看记忆：/memory 概览 · /memory list [类型] 列出 · /memory <关键词> 搜索",
)
async def _memory(conversation_key: str, agent: Any, args: str = "") -> str:
    stats = await agent.memory.get_stats()
    if not stats.get("enabled"):
        return "记忆系统未启用（config `[memory] enabled` 或 CONNECTCLAW_MEMORY_ENABLED）。"

    args = args.strip()

    # 概览：统计 + 最重要的若干条
    if not args:
        lines = [
            "**记忆统计**",
            f"- 🧠 语义 semantic：{stats.get('semantic', 0)}",
            f"- 📅 情景 episodic：{stats.get('episodic', 0)}",
            f"- 🔧 程序 procedural：{stats.get('procedural', 0)}",
            f"- 合计 {stats.get('total', 0)} 条 · DB {stats.get('db_size_kb', 0)} KB",
        ]
        top = await agent.memory.list_memories(limit=10)
        if top:
            lines.append("")
            lines.append(
                "**最重要的记忆**（`/memory list` 看更多 · `/memory <关键词>` 搜索）"
            )
            lines.extend(_fmt_memory(m) for m in top)
        return "\n".join(lines)

    parts = args.split(maxsplit=1)
    head = parts[0].lower()

    # 列出：/memory list [semantic|episodic|procedural]
    if head == "list":
        type_filter = parts[1].strip().lower() if len(parts) > 1 else None
        mems = await agent.memory.list_memories(memory_type=type_filter, limit=25)
        if not mems:
            return f"没有「{type_filter or '任何'}」类型的记忆。"
        title = f"**记忆列表**（{type_filter or '全部'} · {len(mems)} 条）"
        return "\n".join([title] + [_fmt_memory(m) for m in mems])

    # 搜索：/memory search <词> 或直接 /memory <词>
    query = parts[1] if head == "search" and len(parts) > 1 else args
    mems = await agent.memory.list_memories(query=query, limit=15)
    if not mems:
        return f"没有找到与「{query}」相关的记忆。"
    title = f"**搜索「{query}」**（{len(mems)} 条）"
    return "\n".join([title] + [_fmt_memory(m, show_detail=True) for m in mems])


@register("/dream", "整合记忆（做梦）——后台执行，完成后通知")
async def _dream(conversation_key: str, agent: Any, args: str = "") -> str:
    if not agent.memory.enabled:
        return "记忆系统未启用。"

    # Dreaming can take a while (decay sweep + optional LLM consolidation).
    # Reply immediately, run it in the background, then push the result — so
    # the user always gets feedback both at start and on completion/failure.
    channel = getattr(agent, "_channel", None)
    memory = agent.memory
    model = agent._model
    api_key = agent._config.llm.api_key or None

    async def _run() -> None:
        try:
            result = await memory.dream(model, api_key=api_key, force=True)
            if result is None:
                msg = "记忆系统未初始化，无法做梦。"
            else:
                msg = (
                    "💤 **做梦完成**\n"
                    f"- 衰减 {result['decayed']} · 强化 {result['strengthened']} · "
                    f"新语义 {result['new_semantic']} · 合并 {result['merged']} · "
                    f"清理 {result['cleaned']}"
                )
        except Exception as e:
            logger.error("Dream failed: %s", e)
            msg = f"做梦出错：{e}"
        if channel is not None:
            try:
                await channel.send_message(conversation_key, msg)
            except Exception as e:
                logger.debug("Failed to push dream result: %s", e)

    asyncio.create_task(_run())
    return "💤 开始整合记忆（做梦）……完成后会告诉你。"


@register(
    "/forget",
    "删除记忆：/forget 全部 · /forget <词> 按词 · /forget id <id> 按ID · /forget type <类型>",
)
async def _forget(conversation_key: str, agent: Any, args: str = "") -> str:
    if not agent.memory.enabled:
        return "记忆系统未启用。"
    args = args.strip()

    # /forget  →  clear all
    if not args:
        count = await agent.memory.clear_all()
        return f"已清空 {count} 条记忆。"

    parts = args.split(maxsplit=1)
    head = parts[0].lower()
    tail = parts[1].strip() if len(parts) > 1 else ""

    # /forget id <id>  →  delete one by id (bypasses persona protection)
    if head == "id" and tail:
        ok = await agent.memory.forget_by_id(tail)
        return (
            f"已删除记忆 `{tail}`。" if ok
            else f"未找到 id 为 `{tail}` 的记忆。"
        )

    # /forget type <semantic|episodic|procedural>
    if head == "type" and tail:
        count = await agent.memory.forget_by_type(tail)
        if count == 0:
            return f"没有「{tail}」类型的可删除记忆（可能不存在，或被 persona 保护）。"
        return (
            f"已删除 {count} 条「{tail}」记忆。"
            "（persona 级高重要度记忆已跳过，需用 `/forget id <id>` 单独删）"
        )

    # /forget <keyword>  →  delete by keyword match
    count = await agent.memory.forget_by_keyword(args)
    if count == 0:
        return (
            f"没有匹配「{args}」的可删除记忆（可能不存在，或被 persona 保护，"
            "用 `/memory {args}` 先查看）。"
        )
    return (
        f"已删除 {count} 条匹配「{args}」的记忆。"
        "（persona 级高重要度记忆已跳过，需用 `/forget id <id>` 单独删）"
    )


@register("/restart", "重启 ConnectClaw 进程（优雅退出，由守护进程自动拉起）")
async def _restart(conversation_key: str, agent: Any, args: str = "") -> str:
    """设置重启事件，main.py 检测到后执行优雅关闭。"""
    if hasattr(agent, '_restart_event') and agent._restart_event:
        agent._restart_event.set()
        return "♻️ 正在重启 ConnectClaw 进程，请稍候…"
    return "❌ 重启功能未启用（_restart_event 未设置）"


# ── Memory formatting helpers ──────────────────────────────────


_TYPE_EMOJI = {"semantic": "🧠", "episodic": "📅", "procedural": "🔧"}


def _fmt_memory(m: Any, *, show_detail: bool = False) -> str:
    """Render one memory entry as a human-readable markdown line."""
    tval = m.type.value if hasattr(m.type, "value") else str(m.type)
    emoji = _TYPE_EMOJI.get(tval, "•")
    age = _relative_age(getattr(m, "last_accessed", 0.0))
    line = (
        f"- {emoji} `{m.id}` {m.content}  "
        f"_(重要 {m.importance:.1f} · 强度 {m.strength:.1f} · {age}前)_"
    )
    if show_detail and getattr(m, "detail", None):
        line += f"\n    ↳ {m.detail[:180]}"
    return line


def _relative_age(ts: float) -> str:
    if not ts:
        return "?"
    delta = max(0.0, time.time() - ts)
    if delta < 3600:
        return f"{int(delta / 60)}分钟"
    if delta < 86400:
        return f"{int(delta / 3600)}小时"
    return f"{int(delta / 86400)}天"
