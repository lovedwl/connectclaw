"""Interactive /model picker — a two-level Feishu CardKit 2.0 selector.

Level 1: one button per PROVIDER (provider ≈ key identity — same base_url
with different keys are different providers), the active one highlighted.
Level 2: that provider's profiles; clicking one hot-switches immediately
(``CodingAgent.apply_model_profile`` — same path as ``/model set``).

Clicks come back through ``channel.handle_card_action`` (value kind =
``model_switch``) and are dispatched to :func:`on_model_card_action`, wired
in main.py. The card is updated in place at every level transition, so one
picker card walks provider → model → result without spamming the chat.
"""

from __future__ import annotations

import uuid
from typing import Any

from connectclaw.logging import get_logger
from connectclaw.model_registry import ModelProfile

logger = get_logger(__name__)

KIND = "model_switch"

# Grey note under every level — sets expectations + documents the trust model.
_NOTE = "<font color='grey'>点击即热切换（全部会话立即生效，写入 config.toml），无需重启 · /model list 看文本清单</font>"


def _render_nonce() -> str:
    """Per-render random tag baked into every button value.

    The SDK dedups card clicks for 12h on ``message_id + operator + value``
    (WS-redelivery guard). The picker reuses ONE message (in-place updates),
    so a value like ``{"stage": "providers"}`` recurs across renders —
    重新选择 → 返回 → 再展开 would be silently dropped as duplicates. A fresh
    nonce per render keeps genuine re-clicks distinct while a redelivered
    click (same render, same JSON) still dedups.
    """
    return uuid.uuid4().hex[:8]


def group_by_provider(profiles: list[ModelProfile]) -> list[tuple[str, list[ModelProfile]]]:
    """Group profiles by display provider, stable order: insertion of first
    member, active-independent (the caller highlights the active one)."""
    groups: dict[str, list[ModelProfile]] = {}
    for p in profiles:
        groups.setdefault(p.display_provider(), []).append(p)
    return list(groups.items())


def _button(text: str, value: dict, style: str = "default", nonce: str = "") -> dict:
    v = dict(value)
    if nonce:
        v["n"] = nonce
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": text},
        "type": style,
        "value": v,
    }


def _row(*buttons: dict) -> dict:
    """CardKit 2.0 has no action container — a button row is a column_set."""
    return {
        "tag": "column_set",
        "columns": [
            {"tag": "column", "elements": [b]} for b in buttons
        ],
    }


def _card(title: str, template: str, elements: list[dict]) -> dict:
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": title},
                   "template": template},
        "body": {"elements": elements},
    }


def _is_active(p: ModelProfile, active: Any) -> bool:
    # resolved_api_key: the active entry carries the EXPANDED key, a registry
    # profile may hold "$ENV" — comparing raw values would drop the ✅ marker.
    return (active is not None
            and p.base_url == getattr(active, "base_url", None)
            and p.model_id == getattr(active, "model_id", None)
            and p.resolved_api_key() == getattr(active, "api_key", None))


def provider_level_card(agent: Any) -> dict:
    """Level 1: current model + one button per provider (+ 取消)."""
    profiles = agent.list_model_profiles()
    active = agent.entry_model_profile()
    active_label = f"{active.display_provider() if active else '?'}/{getattr(active, 'model_id', '?')}"
    groups = group_by_provider(profiles)
    nonce = _render_nonce()

    elements: list[dict] = [
        {"tag": "markdown",
         "content": f"**当前模型：{active_label}**\n选择模型供应商（点击展开该 key 下的模型）："},
        {"tag": "hr"},
    ]
    for prov, ps in groups:
        has_active = any(_is_active(p, active) for p in ps)
        elements.append(_row(_button(
            f"{prov} · {len(ps)}个" + ("（当前）" if has_active else ""),
            {"kind": KIND, "stage": "models", "provider": prov},
            "primary" if has_active else "default",
            nonce=nonce,
        )))
    elements.append({"tag": "hr"})
    elements.append(_row(_button("取消", {"kind": KIND, "stage": "cancel"}, nonce=nonce)))
    elements.append({"tag": "markdown", "content": _NOTE})
    return _card("模型切换", "blue", elements)


def model_level_card(agent: Any, provider: str) -> dict:
    """Level 2: one button per profile in this provider (+ 返回/取消)."""
    profiles = agent.list_model_profiles()
    active = agent.entry_model_profile()
    ps = [p for p in profiles if p.display_provider() == provider]
    nonce = _render_nonce()

    elements: list[dict] = [
        {"tag": "markdown",
         "content": f"**{provider}** · {len(ps)} 个模型\n点击模型直接切换："},
        {"tag": "hr"},
    ]
    for p in ps:
        is_cur = _is_active(p, active)
        label = p.model_id + ("（当前）" if is_cur else "")
        elements.append(_row(_button(
            label, {"kind": KIND, "stage": "switch", "name": p.name},
            "primary" if is_cur else "default",
            nonce=nonce,
        )))
    elements.append({"tag": "hr"})
    elements.append(_row(
        _button("‹ 返回", {"kind": KIND, "stage": "providers"}, nonce=nonce),
        _button("取消", {"kind": KIND, "stage": "cancel"}, nonce=nonce),
    ))
    elements.append({"tag": "markdown", "content": _NOTE})
    return _card(f"模型切换 · {provider}", "blue", elements)


def result_card(text: str, ok: bool | None = True) -> dict:
    """Terminal state of the picker: switch outcome (+ 重新选择)."""
    template = {True: "green", False: "red", None: "grey"}[ok]
    nonce = _render_nonce()
    elements: list[dict] = [{"tag": "markdown", "content": text}, {"tag": "hr"}]
    if ok is not False:
        elements.append(_row(_button(
            "‹ 重新选择", {"kind": KIND, "stage": "providers"}, nonce=nonce)))
    elements.append({"tag": "markdown", "content": _NOTE})
    return _card("模型切换", template, elements)


async def on_model_card_action(agent: Any, value: dict, chat_id: str, message_id: str) -> None:
    """Handle one picker click: transition the card / perform the hot switch.

    Runs on the SDK background loop (same loop as message handling), so it
    may touch the agent freely. All failures land in the card itself —
    a stuck "思考中"-style silent card is exactly what this replaces.
    """
    channel = getattr(agent, "_channel", None)
    if channel is None or not message_id:
        logger.error("model_card action without channel/message_id: %s", value)
        return

    stage = value.get("stage", "")
    try:
        if stage == "cancel":
            await channel.update_card(message_id, result_card("已取消，模型未变更。", ok=None))
        elif stage == "providers":
            await channel.update_card(message_id, provider_level_card(agent))
        elif stage == "models":
            await channel.update_card(
                message_id, model_level_card(agent, value.get("provider", "")))
        elif stage == "switch":
            name = value.get("name", "")
            p = agent.get_model_profile(name)
            if p is None:
                await channel.update_card(
                    message_id,
                    result_card(f"❌ profile **{name}** 不存在（可能已被删除），请重新选择。", ok=False))
                return
            result = await agent.apply_model_profile(p)
            await channel.update_card(message_id, result_card(result or f"✅ 已切换到 `{p.model_id}`", ok=True))
        else:
            await channel.update_card(message_id, result_card(f"未知的卡片操作：{stage}", ok=False))
    except Exception as e:  # noqa: BLE001
        logger.error("model_card action failed (%s): %s", stage, e)
        try:
            await channel.update_card(message_id, result_card(f"❌ 操作失败：{e}", ok=False))
        except Exception:
            pass
