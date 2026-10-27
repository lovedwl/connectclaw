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
from connectclaw import model_discovery
from connectclaw.model_discovery import fmt_ctx
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
    elements.append(_row(_button(
        "🔍 自动发现新模型（查列表+上下文）",
        {"kind": KIND, "stage": "discover", "provider": provider},
        "primary", nonce=nonce,
    )))
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


# ── Auto-discovery (2026-10) ─────────────────────────────────────
# Level 3: after selecting a provider, one click fetches {base}/models with
# that provider's key, parses per-model limits, and lets the user add new
# profiles / refresh existing ones. Discovery state lives in a tiny
# message_id-keyed cache (the card is updated in place, so the click payload
# carries only ids, not the fetched data).

_DISCOVERY_CACHE: dict[str, dict] = {}
_DISCOVERY_CACHE_MAX = 20
_DISCOVERY_MAX_ROWS = 24          # per-card display cap (payload keeps all)
_DISCOVERY_PROBE_CAP = 8          # context-limit probes per click round


def _cache_put(message_id: str, payload: dict) -> None:
    _DISCOVERY_CACHE[message_id] = payload
    while len(_DISCOVERY_CACHE) > _DISCOVERY_CACHE_MAX:
        _DISCOVERY_CACHE.pop(next(iter(_DISCOVERY_CACHE)))


def _cache_get(message_id: str) -> dict | None:
    return _DISCOVERY_CACHE.get(message_id)


def _proxy_of(agent: Any, base_url: str) -> str | None:
    """Proxy for this endpoint — per-host routing, not one global URL."""
    cfg = getattr(agent, "_config", None)
    proxy = getattr(cfg, "proxy", None)
    if proxy is None:
        return None
    resolver = getattr(proxy, "for_url", None)
    if callable(resolver):
        return resolver(base_url) or None
    return getattr(proxy, "url", "") or None


def _active_of(agent: Any) -> str | None:
    """当前激活 profile 的名字：config.llm 与注册表比对定位。

    CodingAgent 并没有 active_model_name 属性（2026-10 发现，之前的
    getattr 护栏是哑的），这里用 entry_model_profile() 的 model_id+host
    匹配注册表；getattr 保留作测试桩回退。"""
    try:
        llm = agent.entry_model_profile()
        for p in agent.list_model_profiles():
            if (p.model_id == llm.model_id
                    and model_discovery._host(p.base_url)
                    == model_discovery._host(llm.base_url)):
                return p.name
    except Exception:  # noqa: BLE001  FakeAgent 等没有 entry_model_profile
        pass
    return getattr(agent, "active_model_name", None)


def discovery_card(payload: dict, note: str = "") -> dict:
    """Level 3: discovered models with per-model 添加/更新 buttons.

    2026-10 排版重做（移动端反馈「排版垃圾」）：
    - 行内按钮只放 emoji（➕/🔄），图例放头部——旧标签「➕ 添加」
      在 3:1 列宽里被截成「＋…」。
    - 底部三按钮平分被挤成「全…/探…/…」：拆成三行，主 CTA 独占整行。
    - 已知上下文的模型排前面（977k 的先看见）。
    """
    gone = [i for i in payload["items"] if i.get("gone")]
    items = sorted(
        [i for i in payload["items"] if not i.get("gone")],
        key=lambda i: (not i.get("ctx"), -(i.get("ctx") or 0)),
    )
    shown = items[:_DISCOVERY_MAX_ROWS]
    nonce = _render_nonce()
    dead = bool(payload.get("dead_key"))
    if dead:
        head = (f"**{payload['provider']}** \u00b7 \U0001f6ab "
                f"<font color='red'>key \u5df2\u5931\u6548\u2014\u2014 "
                f"{len(gone)} \u4e2a profile \u4e0d\u53ef\u7528</font>")
    else:
        head = (f"**{payload['provider']}** \u00b7 \u53d1\u73b0 {len(items)} \u4e2a\u6a21\u578b"
                + (f" \u00b7 <font color='red'>\U0001f5d1 \u5df2\u4e0b\u67b6 {len(gone)}</font>"
                   if gone else "")
                + f"\n<font color='grey'>`{payload['endpoint']}` \u00b7 ctx/\u8f93\u51fa\uff0c"
                  "?=\u672a\u66b4\u9732 \u00b7 \u2795 \u6dfb\u52a0 \u00b7 \U0001f504 \u66f4\u65b0 "
                  "\u00b7 \U0001f5d1 \u5220\u9664</font>")
    elements: list[dict] = [
        {"tag": "markdown", "content": head},
        {"tag": "hr"},
    ]
    for item in shown:
        label = f"`{item['id']}`  {fmt_ctx(item.get('ctx'))}/{fmt_ctx(item.get('mt'))}"
        if item.get("exists"):
            label += f"\n<font color='grey'>已有 {item['exists']}</font>"
            btn = _button("🔄",
                          {"kind": KIND, "stage": "update_model",
                           "name": item["exists"], "id": item["id"]},
                          "default", nonce=nonce)
        else:
            btn = _button("➕",
                          {"kind": KIND, "stage": "add_model", "id": item["id"]},
                          "default", nonce=nonce)
        elements.append({
            "tag": "column_set",
            "columns": [
                {"tag": "column", "weight": 3,
                 "elements": [{"tag": "markdown", "content": label}]},
                {"tag": "column", "weight": 1, "elements": [btn]},
            ],
        })
    if len(items) > _DISCOVERY_MAX_ROWS:
        elements.append({"tag": "markdown", "content":
                         f"<font color='grey'>仅显示前 {_DISCOVERY_MAX_ROWS} 个；"
                         f"其余 {len(items) - _DISCOVERY_MAX_ROWS} 个可用 "
                         f"`/model sync {payload['provider']}` 文本处理</font>"})
    if gone:
        elements.append({"tag": "markdown", "content":
                         ("🚫 <font color='red'>**key \u5df2\u5931\u6548\uff0c\u4ee5\u4e0b profile \u5747\u4e0d\u53ef\u7528**\uff0c\u53ef\u6e05\u7406\uff1a</font>" if dead
                          else "\u26a0\ufe0f <font color='red'>**\u672c\u5730\u5df2\u6709\u4f46\u7f51\u5173\u5df2\u4e0d\u63d0\u4f9b**"
                                "\uff08\u4e0b\u67b6\u6216\u8be5 key \u65e0\u6743\u9650\uff09\uff0c\u53ef\u6e05\u7406\uff1a</font>")})
        for item in gone:
            label = (f"`{item['id']}`\n<font color='grey'>\u5df2\u6709 "
                     f"{item['exists']} \u00b7 ctx {fmt_ctx(item.get('ctx'))}</font>")
            elements.append({
                "tag": "column_set",
                "columns": [
                    {"tag": "column", "weight": 3,
                     "elements": [{"tag": "markdown", "content": label}]},
                    {"tag": "column", "weight": 1,
                     "elements": [_button("\U0001f5d1",
                                          {"kind": KIND, "stage": "delete_model",
                                           "name": item["exists"]},
                                          "danger", nonce=nonce)]},
                ],
            })
        elements.append(_row(_button(
            f"🗑 全部清理（{len(gone)}）",
            {"kind": KIND, "stage": "delete_all_gone"},
            "danger", nonce=nonce)))
    elements.append({"tag": "hr"})
    n_unknown = sum(1 for i in items if not i.get("ctx"))
    if n_unknown:
        elements.append(_row(_button(
            f"🔎 探测未知上限（{min(n_unknown, _DISCOVERY_PROBE_CAP)}）",
            {"kind": KIND, "stage": "probe_unknown"},
            "default", nonce=nonce)))
    if not dead:
        elements.append(_row(_button(
            "\u2795 \u5168\u90e8\u6dfb\u52a0/\u66f4\u65b0", {"kind": KIND, "stage": "add_all"},
            "primary", nonce=nonce)))
    elements.append(_row(_button(
        "‹ 返回模型列表", {"kind": KIND, "stage": "models",
                             "provider": payload["provider"]},
        "default", nonce=nonce)))
    if note:
        elements.append({"tag": "markdown",
                         "content": f"<font color='grey'>{note}</font>"})
    elements.append({"tag": "markdown", "content": _NOTE})
    return _card(f"模型发现 · {payload['provider']}", "turquoise", elements)


async def _stage_discover(agent, channel, value: dict, message_id: str) -> None:
    provider = value.get("provider", "")
    profiles = [p for p in agent.list_model_profiles()
                if p.display_provider() == provider]
    if not profiles:
        await channel.update_card(
            message_id, result_card(f"❌ 没有「{provider}」的 profile，无法借用其 key 查询。", ok=False))
        return
    seed = profiles[0]
    await channel.update_card(
        message_id,
        result_card(f"🔍 正在查询 **{provider}** 的模型列表（{seed.base_url}）…", ok=None))
    try:
        entries, endpoint = await model_discovery.fetch_model_list(
            seed.base_url, seed.resolved_api_key(), _proxy_of(agent, seed.base_url))
    except model_discovery.AuthError as e:
        # key 已死：该 provider 全部 profile 都不可用 → 转入清理模式，
        # 复用 gone 清单 + 🗑 逐个删（正在使用的模型受同样护栏）。
        dead_items = [{"id": p.model_id or "?", "ctx": p.context_window or None,
                       "mt": p.max_tokens or None, "exists": p.name, "gone": True}
                      for p in profiles]
        payload = {"provider": provider, "base_url": seed.base_url,
                   "raw_key": seed.api_key, "resolved_key": seed.resolved_api_key(),
                   "endpoint": "", "items": dead_items, "dead_key": True}
        _cache_put(message_id, payload)
        await channel.update_card(
            message_id,
            discovery_card(payload,
                           note=f"🚫 {e} \u2014\u2014 \u53ef\u9010\u4e2a \U0001f5d1 \u6e05\u7406\uff0c\u6216\u7528\u65b0 key \u91cd\u65b0\u6dfb\u52a0\u3002"))
        return
    except model_discovery.DiscoveryError as e:
        await channel.update_card(message_id, result_card(f"\u274c \u53d1\u73b0\u5931\u8d25\uff1a{e}", ok=False))
        return
    if not entries:
        await channel.update_card(
            message_id,
            result_card(f"⚠️ `{endpoint}/models` 返回空列表（该网关不暴露模型清单）。"
                        f"请用 `/model add` 手动添加。", ok=None))
        return
    items = []
    for e in entries:
        ex = model_discovery.match_existing(profiles, seed.base_url, e["id"])
        items.append({"id": e["id"], "ctx": e["max_input"], "mt": e["max_output"],
                      "exists": ex.name if ex else None})
    # \u4e0b\u67b6\u68c0\u6d4b\uff1a\u540c host \u7684\u672c\u5730 profile \u5176 model_id \u5df2\u4e0d\u5728\u4e0a\u6e38\u5217\u8868\u91cc
    # \uff08\u5217\u8868\u6309 key \u6743\u9650\u8fc7\u6ee4\uff0ckey \u7528\u4e0d\u4e86\u540c\u6837\u7b49\u4e8e\u4e0d\u53ef\u7528\uff09\u2192 \u6807 gone \u4f9b\u4e00\u952e\u6e05\u7406\u3002
    upstream_ids = {e["id"] for e in entries}
    seed_host = model_discovery._host(seed.base_url)
    for p in profiles:
        if (p.model_id and seed_host
                and model_discovery._host(p.base_url) == seed_host
                and p.model_id not in upstream_ids):
            items.append({"id": p.model_id, "ctx": p.context_window or None,
                          "mt": p.max_tokens or None,
                          "exists": p.name, "gone": True})
    payload = {"provider": provider, "base_url": seed.base_url,
               "raw_key": seed.api_key, "resolved_key": seed.resolved_api_key(),
               "endpoint": endpoint, "items": items}
    _cache_put(message_id, payload)
    await channel.update_card(message_id, discovery_card(payload))


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
        elif stage == "discover":
            await _stage_discover(agent, channel, value, message_id)
        elif stage == "add_model":
            payload = _cache_get(message_id)
            item = next((i for i in (payload or {}).get("items", [])
                         if i["id"] == value.get("id")), None)
            if payload is None or item is None:
                await channel.update_card(
                    message_id, result_card("⚠️ 发现结果已过期，请重新 `/model` 选择供应商。", ok=None))
                return
            if item.get("exists"):
                await channel.update_card(
                    message_id,
                    result_card(f"ℹ️ `{item['id']}` 已有 profile **{item['exists']}**，请用「🔄 更新」。", ok=None))
                return
            p = ModelProfile(
                name="", base_url=payload["base_url"], model_id=item["id"],
                api_key=payload["raw_key"],
                context_window=item.get("ctx") or model_discovery.DEFAULT_CONTEXT_WINDOW,
                max_tokens=item.get("mt") or model_discovery.DEFAULT_MAX_TOKENS,
                provider=payload["provider"])
            name = agent.save_model_profile(p)
            await channel.update_card(message_id, result_card(
                f"✅ 已添加 **{name}**（`{item['id']}`）\n"
                f"上下文 {fmt_ctx(item.get('ctx') or model_discovery.DEFAULT_CONTEXT_WINDOW)} · "
                f"输出 {fmt_ctx(item.get('mt') or model_discovery.DEFAULT_MAX_TOKENS)}\n"
                f"可 `/model test {name}` 实测后 `/model set {name}` 激活。", ok=True))
        elif stage == "update_model":
            payload = _cache_get(message_id)
            item = next((i for i in (payload or {}).get("items", [])
                         if i["id"] == value.get("id")), None)
            cur = agent.get_model_profile(value.get("name", ""))
            if payload is None or item is None or cur is None:
                await channel.update_card(
                    message_id, result_card("⚠️ 发现结果已过期或 profile 不存在，请重新 `/model`。", ok=None))
                return
            changed = []
            if item.get("ctx"):
                cur.context_window = item["ctx"]
                changed.append(f"上下文→{fmt_ctx(item['ctx'])}")
            if item.get("mt"):
                cur.max_tokens = item["mt"]
                changed.append(f"输出→{fmt_ctx(item['mt'])}")
            if not changed:
                await channel.update_card(
                    message_id,
                    result_card(f"❓ `{item['id']}` 未暴露上下文/输出上限，没有可更新的字段。\n"
                                f"可点「🔎 探测上限」实测，或 `/model edit {cur.name}` 手填。", ok=None))
                return
            agent.save_model_profile(cur)
            await channel.update_card(message_id, result_card(
                f"🔄 已更新 **{cur.name}**（`{item['id']}`）：{'，'.join(changed)}", ok=True))
        elif stage == "add_all":
            payload = _cache_get(message_id)
            if payload is None:
                await channel.update_card(
                    message_id, result_card("⚠️ 发现结果已过期，请重新 `/model` 选择供应商。", ok=None))
                return
            added, updated, skipped = [], [], []
            for item in payload["items"]:
                if item.get("gone"):
                    continue  # \u5df2\u4e0b\u67b6\u6a21\u578b\u4e0d\u53c2\u4e0e\u6279\u91cf\u540c\u6b65
                if item.get("exists"):
                    cur = agent.get_model_profile(item["exists"])
                    if cur is None:
                        skipped.append(f"{item['id']}（已有 profile 不存在）")
                        continue
                    if item.get("ctx") or item.get("mt"):
                        if item.get("ctx"):
                            cur.context_window = item["ctx"]
                        if item.get("mt"):
                            cur.max_tokens = item["mt"]
                        agent.save_model_profile(cur)
                        updated.append(item["id"])
                    else:
                        skipped.append(f"{item['id']}（未暴露上限）")
                else:
                    p = ModelProfile(
                        name="", base_url=payload["base_url"], model_id=item["id"],
                        api_key=payload["raw_key"],
                        context_window=item.get("ctx") or model_discovery.DEFAULT_CONTEXT_WINDOW,
                        max_tokens=item.get("mt") or model_discovery.DEFAULT_MAX_TOKENS,
                        provider=payload["provider"])
                    added.append(agent.save_model_profile(p))
            text = (f"✅ 批量同步完成（{payload['provider']}，共 {len(payload['items'])} 个）：\n"
                    f"· 新增 {len(added)}：{'、'.join(added) or '无'}\n"
                    f"· 更新 {len(updated)}：{'、'.join(updated) or '无'}\n"
                    f"· 跳过 {len(skipped)}：{'、'.join(skipped) or '无'}\n"
                    "新模型可 `/model test <name>` 实测后 `/model set <name>` 激活。")
            await channel.update_card(message_id, result_card(text, ok=True))
        elif stage == "probe_unknown":
            payload = _cache_get(message_id)
            if payload is None:
                await channel.update_card(
                    message_id, result_card("⚠️ 发现结果已过期，请重新 `/model` 选择供应商。", ok=None))
                return
            unknowns = [i for i in payload["items"] if not i.get("ctx")]
            if not unknowns:
                await channel.update_card(
                    message_id, result_card("所有模型都已有上下文长度。", ok=None))
                return
            batch = unknowns[:_DISCOVERY_PROBE_CAP]
            await channel.update_card(
                message_id,
                result_card(f"🔎 正在探测 {len(batch)} 个模型的上下文上限"
                            f"（发送超大 prompt 触发 400，每个可能 10~60 秒）…", ok=None))
            for item in batch:
                limit, note = await model_discovery.probe_context_window(
                    payload["base_url"], payload["resolved_key"], item["id"], _proxy_of(agent, payload["base_url"]))
                if limit:
                    item["ctx"] = limit
                item["probe_note"] = note
            _cache_put(message_id, payload)
            tail = ""
            if len(unknowns) > len(batch):
                tail = f"；本轮最多 {len(batch)} 个，剩余 {len(unknowns) - len(batch)} 个可再点一次"
            await channel.update_card(
                message_id,
                discovery_card(payload, note=f"探测完成{tail}；? = 未解析到"))

        elif stage == "delete_model":
            payload = _cache_get(message_id)
            name = value.get("name", "")
            if payload is None:
                await channel.update_card(
                    message_id, result_card("\u26a0\ufe0f \u53d1\u73b0\u7ed3\u679c\u5df2\u8fc7\u671f\uff0c\u8bf7\u91cd\u65b0 `/model` \u3002", ok=None))
                return
            # \u53ea\u5141\u8bb8\u5220\u300c\u4e0b\u67b6\u68c0\u6d4b\u300d\u6807\u51fa\u7684 profile\uff0c\u9632\u6784\u9020 value \u4efb\u610f\u5220
            target = next((i for i in payload["items"]
                           if i.get("gone") and i.get("exists") == name), None)
            if target is None:
                await channel.update_card(
                    message_id,
                    result_card(f"\u26a0\ufe0f `{name}` \u4e0d\u5728\u5df2\u4e0b\u67b6\u6e05\u5355\u4e2d\uff08\u6216\u6e05\u5355\u5df2\u8fc7\u671f\uff09\uff0c\u8bf7\u91cd\u65b0\u53d1\u73b0\u3002", ok=False))
                return
            if name == _active_of(agent):
                await channel.update_card(
                    message_id,
                    result_card(f"\u274c **{name}** \u6b63\u5728\u4f7f\u7528\u4e2d\uff0c\u5148\u5207\u6362\u5230\u522b\u7684\u6a21\u578b\u518d\u5220\u9664\u3002", ok=False))
                return
            if not agent.delete_model_profile(name):
                await channel.update_card(
                    message_id,
                    result_card(f"\u274c \u5220\u9664\u5931\u8d25\uff1a\u6ca1\u6709\u540d\u4e3a {name} \u7684 profile\u3002", ok=False))
                return
            payload["items"] = [i for i in payload["items"] if i is not target]
            _cache_put(message_id, payload)
            await channel.update_card(
                message_id,
                discovery_card(payload, note=f"\U0001f5d1 \u5df2\u5220\u9664 **{name}**\uff08`{target['id']}`\uff09"))
        elif stage == "delete_all_gone":
            payload = _cache_get(message_id)
            if payload is None:
                await channel.update_card(
                    message_id, result_card("⚠️ 发现结果已过期，请重新 `/model`。", ok=None))
                return
            active = _active_of(agent)
            deleted, kept = [], []
            for it in [i for i in payload["items"] if i.get("gone")]:
                nm = it.get("exists")
                if not nm:
                    continue
                if nm == active:
                    kept.append(nm)
                    continue
                if agent.delete_model_profile(nm):
                    deleted.append(nm)
            payload["items"] = [i for i in payload["items"]
                                if not (i.get("gone")
                                        and i.get("exists") in deleted)]
            _cache_put(message_id, payload)
            note = f"🗑 已清理 {len(deleted)} 个：{'、'.join(deleted) or '无'}"
            if kept:
                note += f"；正在使用的 **{'、'.join(kept)}** 已保留"
            await channel.update_card(
                message_id, discovery_card(payload, note=note))
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
