"""
Slash commands for ConnectClaw.

Commands are exact-matched against the stripped message text.
Add new commands by adding entries to COMMANDS below.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable


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

    cmd = COMMANDS.get(stripped)
    if cmd is None:
        return _help_text(stripped)

    return await cmd.handler(conversation_key, agent)


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
        async def _foo(conversation_key, agent) -> str:
            return "done"
    """
    def decorator(fn: CommandFn) -> CommandFn:
        COMMANDS[name] = Command(name=name, description=description, handler=fn)
        return fn
    return decorator


# ── Internals ──────────────────────────────────────────────────


CommandFn = Callable[[str, Any], Awaitable[str]]


class Command:
    __slots__ = ("name", "description", "handler")

    def __init__(self, *, name: str, description: str, handler: CommandFn):
        self.name = name
        self.description = description
        self.handler = handler


COMMANDS: dict[str, Command] = {}


# ── Built-in Commands ──────────────────────────────────────────


@register("/stop", "interrupt the running agent loop")
async def _stop(conversation_key: str, agent: Any) -> str:
    agent.abort(conversation_key)
    return "⏹ Interrupted."


@register("/new", "start a fresh conversation (clear context)")
async def _new(conversation_key: str, agent: Any) -> str:
    await agent.new_session(conversation_key)
    return "🆕 Fresh conversation started."
