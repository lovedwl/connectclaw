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
        full_id = await agent.memory.forget_by_id(tail)
        return (
            f"已删除记忆 `{full_id}`。" if full_id
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


@register("/whoami", "查看自己的 open_id（填入 [bash] operator_open_ids 即可加入 !/bash 直连白名单）")
async def _whoami(conversation_key: str, agent: Any, args: str = "") -> str:
    rid = getattr(agent, "last_sender", "") or ""
    if not rid:
        return "❌ 获取不到你的 open_id（只支持在飞书会话中使用）"
    ops: list = []
    cfg = getattr(agent, "_config", None)
    bash_cfg = getattr(cfg, "bash", None) if cfg is not None else None
    if bash_cfg is not None:
        ops = list(bash_cfg.operator_open_ids or [])
    mark = "（已在白名单 ✅）" if rid in ops else "（尚未加入白名单）"
    return (
        f"你的 open_id：`{rid}` {mark}\n"
        f"把它加入 `~/.connectclaw/config.toml` 的 `[bash] operator_open_ids = [...]` "
        f"即可使用 `!命令` / `/bash 命令` 直连执行。"
    )


# ══════════════════════════════════════════════════════════════
# /model —— 模型逃生口（纯配置+直连实测，不依赖当前模型可用）
# ══════════════════════════════════════════════════════════════

_MODEL_BOOLS = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}


def _parse_pairs(args: str) -> dict:
    """Parse ``k=v k2=v2`` pairs with shlex quoting support."""
    import shlex

    out: dict = {}
    for tok in shlex.split(args or ""):
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k.strip()] = v.strip()
    return out


def _profile_from_pairs(pairs: dict) -> dict:
    from connectclaw.model_registry import ModelProfile

    p = ModelProfile(
        name=pairs.get("name", ""),
        base_url=pairs.get("base_url", ""),
        model_id=pairs.get("model_id", ""),
        api_key=pairs.get("api_key", ""),
        desc=pairs.get("desc", ""),
        provider=pairs.get("provider", ""),
    )
    if "reasoning" in pairs:
        p.reasoning = _MODEL_BOOLS.get(pairs["reasoning"].lower(), p.reasoning)
    for k in ("context_window", "max_tokens"):
        if k in pairs:
            try:
                setattr(p, k, int(pairs[k]))
            except ValueError:
                pass
    return p


MODEL_WIZARD_STEPS = ["base_url", "model_id", "api_key"]
_MODEL_WIZARD_PROMPTS = {
    "base_url": "① base_url（OpenAI 兼容端点，如 `https://api.deepseek.com` 或网关地址）：",
    "model_id": "② model_id（模型名，如 `deepseek-v4-flash`）：",
    "api_key": "③ api_key（没有可回复 `无` 跳过）：",
}


async def run_model_wizard_step(agent: Any, state: dict, text: str) -> str:
    """Feed one non-command reply into a pending /model add wizard."""
    from connectclaw.model_registry import ModelProfile

    idx = state.get("step", 0)
    if idx >= len(MODEL_WIZARD_STEPS):
        return "向导已结束，可用 /model 查看。"
    key = MODEL_WIZARD_STEPS[idx]
    value = text.strip()
    if key == "api_key" and value in ("无", "none", "/skip"):
        value = ""
    if key == "base_url" and not value.startswith(("http://", "https://")):
        return "base_url 应以 `http(s)://` 开头，请重试：\n" + _MODEL_WIZARD_PROMPTS[key]
    if key != "api_key" and not value:
        return _MODEL_WIZARD_PROMPTS[key]
    state["fields"][key] = value
    if key == "base_url":
        state["fields"]["base_url"] = value.rstrip("/")
    idx += 1
    state["step"] = idx
    if idx < len(MODEL_WIZARD_STEPS):
        agent.set_wizard(state.get("_conv", ""), state)
        return _MODEL_WIZARD_PROMPTS[MODEL_WIZARD_STEPS[idx]]
    # complete
    agent.pop_wizard(state.get("_conv", ""))
    p = ModelProfile(
        name=state["fields"].get("name", ""),
        base_url=state["fields"].get("base_url", ""),
        model_id=state["fields"].get("model_id", ""),
        api_key=state["fields"].get("api_key", ""),
    )
    if not p.base_url or not p.model_id:
        return "❌ base_url/model_id 不能为空，向导已放弃。"
    name = agent.save_model_profile(p)
    return (
        f"✅ 已保存 profile **{name}**（`{p.model_id}` @ {p.base_url}）。\n"
        f"建议下一步：`/model test {name}` 实测 → 通过后 `/model set {name}` 激活。"
    )


@register(
    "/model",
    "模型逃生口：/model 清单 · /model add 添加(向导) · /model test <name> 实测 · "
    "/model set <name> 激活 · /model edit|rm <name> · /model verify <base_url> · /model cancel",
)
async def _model(conversation_key: str, agent: Any, args: str = "") -> str:
    parts = (args or "").split()
    sub = parts[0].lower() if parts else ""
    rest = args[len(parts[0]):].strip() if parts else ""

    if sub in ("help", "-h", "--help"):
        return _model_help()

    if sub == "":  # /model → interactive picker card
        return await _model_picker_card(agent, conversation_key)

    if sub in ("list", "ls"):  # text list, grouped by provider
        return _model_list(agent)

    if sub == "add":
        pairs = _parse_pairs(rest)
        p = _profile_from_pairs(pairs) if pairs else None
        return await _model_add(agent, conversation_key, p, pairs.get("name", ""))

    if sub == "edit":
        pairs = _parse_pairs(rest)
        name = pairs.pop("name", "") or (parts[1] if len(parts) > 1 else "")
        return await _model_edit(agent, conversation_key, name, pairs)

    if sub == "rm":
        name = rest.strip() or (parts[1] if len(parts) > 1 else "")
        return _model_rm(agent, name)

    if sub == "test":
        name = rest.strip() or ""
        return await _model_test(agent, name)

    if sub == "set":
        name = rest.strip() or (parts[1] if len(parts) > 1 else "")
        return await _model_set(agent, name)

    if sub == "verify":
        url = rest.strip() or (parts[1] if len(parts) > 1 else "")
        return await _model_verify(agent, url)

    if sub == "cancel":
        agent.pop_wizard(conversation_key)
        return "已取消当前向导。"

    return _model_help()


def _model_help() -> str:
    return (
        "**/model —— 模型逃生口**\n"
        "· `/model` 弹出**选择卡片**：供应商（按 key 区分）→ 模型 → 点击即热切换\n"
        "· `/model list` 文本清单（按供应商分组）\n"
        "· `/model add` 交互向导添加；或 `/model add name=x base_url=... model_id=... api_key=... [provider=...]`\n"
        "· `/model test <name>` 对该模型发送最小请求实测（逃生校验）\n"
        "· `/model set <name>` 激活（热切换全部会话 + 写入 config.toml）\n"
        "· `/model edit <name> base_url=... model_id=... [provider=...]` / `/model rm <name>`\n"
        "· `/model verify <base_url>` 查看端点自称的模型列表（*可能不完整，仅供参考*）\n"
        "· `/model cancel` 取消进行中的添加向导\n"
        "无论当前模型是否可用，以上操作都能执行。"
    )


def _model_list(agent: Any) -> str:
    from connectclaw.model_card import group_by_provider

    profiles = agent.list_model_profiles()
    active = agent.entry_model_profile()
    if not profiles:
        lines = ["注册表为空。用 `/model add` 添加第一个 profile（逃生时走向导也行），或 `/model` 弹卡片。"]
    else:
        groups = group_by_provider(profiles)
        lines = [f"模型注册表 · {len(groups)} 个供应商 / {len(profiles)} 个 profile："]
        for prov, ps in groups:
            lines.append(f"\n**{prov}**")
            for p in ps:
                mark = " ✅当前" if (active is not None
                                    and p.base_url == active.base_url
                                    and p.model_id == active.model_id
                                    and p.api_key == active.api_key) else ""
                desc = f" — {p.desc}" if p.desc else ""
                lines.append(f"- **{p.name}** `{p.model_id}`{mark}{desc}")
    if active is not None:
        lines.append(f"\n当前激活：[llm] `{active.model_id}` @ {active.base_url}"
                     f"（{active.display_provider()}）")
    lines.append("用法：/model help")
    return "\n".join(lines)


async def _model_picker_card(agent: Any, conversation_key: str) -> str:
    """/model → interactive picker card. Falls back to the text list if the
    channel can't send cards (channel-less agent, send failure)."""
    from connectclaw.model_card import provider_level_card

    channel = getattr(agent, "_channel", None)
    if channel is None:
        return _model_list(agent)
    msg_id = await channel.send_card(conversation_key, provider_level_card(agent))
    if not msg_id:
        return _model_list(agent)
    return ""  # the card IS the response


async def _model_add(agent: Any, conversation_key: str, partial, name_hint: str = "") -> str:
    if partial is not None and partial.base_url and partial.model_id:
        pname = agent.save_model_profile(partial)
        return (
            f"✅ 已保存 **{pname}**（`{partial.model_id}` @ {partial.base_url}）。\n"
            f"建议：`/model test {pname}` 实测 → `/model set {pname}` 激活。"
        )
    # Wizard mode: no model available? Still works — answers arrive as chat texts.
    fields = {}
    if partial is not None:
        d = partial.__dict__
        fields = {k: d.get(k, "") for k in ("base_url", "model_id", "api_key")}
    if name_hint:
        fields["name"] = name_hint
    state = {"fields": fields, "step": 0, "_conv": conversation_key}
    agent.set_wizard(conversation_key, state)
    return "开始添加模型 profile（向导会逐个收集，随时 `/model cancel` 取消）。\n" + _MODEL_WIZARD_PROMPTS["base_url"]


async def _model_edit(agent: Any, conversation_key: str, name: str, pairs: dict) -> str:
    if not name:
        return "用法：`/model edit <name> base_url=... model_id=... [api_key=...]`"
    cur = agent.get_model_profile(name)
    if cur is None:
        return f"❌ 没有名为 {name} 的 profile（/model 查看）"
    merged = {
        "base_url": pairs.get("base_url", cur.base_url),
        "model_id": pairs.get("model_id", cur.model_id),
        "api_key": pairs.get("api_key", cur.api_key),
        "desc": pairs.get("desc", cur.desc),
        "provider": pairs.get("provider", cur.provider),
        "reasoning": pairs.get("reasoning", str(cur.reasoning).lower()),
        "context_window": pairs.get("context_window", str(cur.context_window)),
        "max_tokens": pairs.get("max_tokens", str(cur.max_tokens)),
    }
    p = _profile_from_pairs(merged)
    p.name = name
    p = _overlay(p, pairs)
    agent.save_model_profile(p)
    return f"✅ 已更新 **{name}** → `{p.model_id}` @ {p.base_url}"


def _overlay(p, pairs: dict):
    if "base_url" in pairs: p.base_url = pairs["base_url"].rstrip("/")
    if "model_id" in pairs: p.model_id = pairs["model_id"]
    if "api_key" in pairs: p.api_key = pairs["api_key"]
    if "desc" in pairs: p.desc = pairs["desc"]
    if "provider" in pairs: p.provider = pairs["provider"]
    if "reasoning" in pairs: p.reasoning = _MODEL_BOOLS.get(pairs["reasoning"].lower(), p.reasoning)
    for k in ("context_window", "max_tokens"):
        if k in pairs:
            try: setattr(p, k, int(pairs[k]))
            except ValueError: pass
    return p


def _model_rm(agent: Any, name: str) -> str:
    if not name:
        return "用法：`/model rm <name>`"
    if agent.delete_model_profile(name):
        return f"🗑️ 已删除 profile **{name}**。"
    return f"❌ 没有名为 {name} 的 profile"


def _active_profile(agent: Any, name: str):
    if not name or name in ("active", "当前", "now", "激活"):
        return agent.entry_model_profile(), None
    p = agent.get_model_profile(name)
    if p is None:
        return None, f"❌ 没有名为 {name} 的 profile（/model 查看）"
    return p, None


async def _model_test(agent: Any, name: str) -> str:
    p, err = _active_profile(agent, name)
    if err:
        return err
    ok, detail = await agent.test_model_profile(p)
    label = name or "当前激活模型"
    return f"🔌 实测 `{label}`（{p.model_id} @ {p.base_url}）：\n{detail}" + ("\n→ 可 `/model set <name>` 激活" if ok else "")


async def _model_set(agent: Any, name: str) -> str:
    if not name:
        return "用法：`/model set <name>`"
    p = agent.get_model_profile(name)
    if p is None:
        return f"❌ 没有名为 {name} 的 profile（/model 查看）"
    if not p.base_url or not p.model_id:
        return f"❌ profile {name} 缺 base_url/model_id，先 `/model edit {name}`"
    try:
        return await agent.apply_model_profile(p)
    except Exception as e:  # noqa: BLE001
        return f"❌ 切换失败：{e}"


async def _model_verify(agent: Any, url: str) -> str:
    import httpx

    if not url:
        return "用法：`/model verify <base_url>`"
    base = url.rstrip("/")
    endpoint = f"{base}/models"
    proxy = getattr(agent, "_config", None) and agent._config.proxy.url or None
    try:
        async with httpx.AsyncClient(
            timeout=12, proxy=proxy or None, trust_env=False,
        ) as c:
            r = await c.get(endpoint)
        if r.status_code != 200:
            return f"⚠️ `{endpoint}` → HTTP {r.status_code}（可能不是 OpenAI 兼容端点或无权限）"
        ids = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict)]
        if not ids:
            return f"`{endpoint}` 返回空模型列表（网关可能不暴露，见 /model help 定位）"
        head = "\n".join(f"- `{m}`" for m in ids[:40])
        more = f"\n…共 {len(ids)} 个（列表可能不完整，可用性以 /model test 为准）" if len(ids) > 40 else ""
        return f"`{base}` 自称提供 {len(ids)} 个模型：\n{head}{more}"
    except Exception as e:  # noqa: BLE001
        return f"❌ 无法访问 `{endpoint}`：{e}"


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
