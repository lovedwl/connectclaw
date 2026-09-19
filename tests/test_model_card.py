"""/model 选择卡片测试：provider 分组、两级卡片、点击分发与热切换。

provider ≈ key 身份：同 base_url 不同 key 是不同供应商（USTC 被封 key
vs 现用 key 必须分开呈现）——用户明确要求卡片按 provider 区分。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from connectclaw.channel.feishu import AuthRequest, FeishuChannel
from connectclaw.config import FeishuConfig
from connectclaw.model_card import (
    group_by_provider,
    model_level_card,
    on_model_card_action,
    provider_level_card,
    result_card,
)
from connectclaw.model_registry import ModelProfile, ModelsStore


def _p(name, model_id, provider="", base_url="https://api.llm.ustc.edu.cn/", key="k1"):
    return ModelProfile(name=name, base_url=base_url, model_id=model_id,
                        api_key=key, provider=provider)


class FakeAgent:
    def __init__(self, profiles, active):
        self._profiles = profiles
        self._active = active
        self.switched: list[str] = []
        self._channel = None

    def list_model_profiles(self):
        return self._profiles

    def entry_model_profile(self):
        return self._active

    def get_model_profile(self, name):
        return next((p for p in self._profiles if p.name == name), None)

    async def apply_model_profile(self, p):
        self.switched.append(p.name)
        return f"✅ 已切换到 `{p.model_id}`（热切换全部会话）"


def _buttons(card):
    out = []
    for el in card["body"]["elements"]:
        if el.get("tag") == "column_set":
            for col in el["columns"]:
                out.extend(col["elements"])
    return out


# ── 分组 ───────────────────────────────────────────────────────


def test_group_by_provider_preserves_order_and_falls_back_to_host():
    ps = [
        _p("a", "m1", provider="USTC 现用key"),
        _p("b", "m2", provider="USTC 现用key"),
        _p("c", "m3", provider="USTC 旧key（已封）", key="k2"),
        _p("d", "m4", provider="", base_url="https://api.deepseek.com/v1"),
    ]
    groups = dict(group_by_provider(ps))
    assert [p.name for p in groups["USTC 现用key"]] == ["a", "b"]
    assert [p.name for p in groups["USTC 旧key（已封）"]] == ["c"]
    assert list(groups) == ["USTC 现用key", "USTC 旧key（已封）", "api.deepseek.com"]


def test_store_provider_roundtrip(tmp_path):
    reg = str(tmp_path / "m.toml")
    ModelsStore(registry_path=reg, config_path=str(tmp_path / "c.toml")).save(
        _p("a", "m1", provider="P1"))
    p = ModelsStore(registry_path=reg, config_path=str(tmp_path / "c.toml")).get("a")
    assert p.provider == "P1"


# ── 卡片结构 ───────────────────────────────────────────────────


def test_provider_level_card_buttons():
    active = _p("a", "m1", provider="USTC 现用key")
    agent = FakeAgent(
        [active, _p("c", "m3", provider="USTC 旧key（已封）", key="k2")], active)
    card = provider_level_card(agent)
    btns = _buttons(card)
    assert all(b["value"]["kind"] == "model_switch" for b in btns)
    models_btn = [b for b in btns if b["value"]["stage"] == "models"]
    assert len(models_btn) == 2
    cur = [b for b in models_btn if "（当前）" in b["text"]["content"]]
    assert len(cur) == 1 and cur[0]["type"] == "primary"
    assert len([b for b in btns if b["value"]["stage"] == "cancel"]) == 1


def test_model_level_card_lists_profiles_with_switch_values():
    active = _p("a", "m1", provider="USTC 现用key")
    agent = FakeAgent([active, _p("b", "m2", provider="USTC 现用key")], active)
    card = model_level_card(agent, "USTC 现用key")
    switch = [b for b in _buttons(card) if b["value"]["stage"] == "switch"]
    assert [b["value"]["name"] for b in switch] == ["a", "b"]
    cur = [b for b in switch if "（当前）" in b["text"]["content"]]
    assert len(cur) == 1 and cur[0]["type"] == "primary"
    stages = [b["value"]["stage"] for b in _buttons(card)]
    assert "providers" in stages and "cancel" in stages


def test_result_card_states():
    ok_card = result_card("好了", ok=True)
    fail_card = result_card("炸了", ok=False)
    assert ok_card["header"]["template"] == "green"
    assert fail_card["header"]["template"] == "red"
    # 成功/取消卡可重选；失败卡不提供重选按钮（值可能已失效，回 /model 重开）
    assert [b["value"]["stage"] for b in _buttons(ok_card)] == ["providers"]
    assert _buttons(fail_card) == []


# ── 点击分发 ───────────────────────────────────────────────────


def _ev(value, chat_id="oc_x", message_id="om_1"):
    return SimpleNamespace(action=SimpleNamespace(value=value, tag="button"),
                           chat_id=chat_id, message_id=message_id)


async def test_click_switch_flows_through_handler():
    ch = FeishuChannel(FeishuConfig(app_id="a", app_secret="s"))
    active = _p("a", "m1", provider="USTC 现用key")
    agent = FakeAgent([active, _p("b", "m2", provider="USTC 现用key")], active)
    agent._channel = ch

    updates: list[tuple[str, dict]] = []

    async def fake_update(message_id, card):
        updates.append((message_id, card))
        return True

    ch.update_card = fake_update

    async def handler(value, chat_id, message_id):
        await on_model_card_action(agent, value, chat_id, message_id)

    ch.set_model_card_handler(handler)

    # 在测试 loop 里直接调（handle_card_action 内部 create_task 依赖运行中的 loop）
    ch.handle_card_action(_ev({"kind": "model_switch", "stage": "switch", "name": "b"}))
    for _ in range(5):
        await asyncio.sleep(0.01)
        if updates:
            break

    assert agent.switched == ["b"]
    assert updates and updates[0][0] == "om_1"
    body = updates[0][1]["body"]["elements"][0]["content"]
    assert "已切换" in body


async def test_click_cancel_and_unknown_profile():
    ch = FeishuChannel(FeishuConfig(app_id="a", app_secret="s"))
    active = _p("a", "m1", provider="P")
    agent = FakeAgent([active], active)
    agent._channel = ch
    updates: list[dict] = []

    async def fake_update(message_id, card):
        updates.append(card)
        return True

    ch.update_card = fake_update
    ch.set_model_card_handler(
        lambda v, c, m: on_model_card_action(agent, v, c, m))

    ch.handle_card_action(_ev({"kind": "model_switch", "stage": "cancel"}))
    for _ in range(5):
        await asyncio.sleep(0.01)
        if updates:
            break
    assert agent.switched == []
    assert "已取消" in updates[0]["body"]["elements"][0]["content"]

    updates.clear()
    ch.handle_card_action(_ev({"kind": "model_switch", "stage": "switch", "name": "ghost"}))
    for _ in range(5):
        await asyncio.sleep(0.01)
        if updates:
            break
    assert "不存在" in updates[0]["body"]["elements"][0]["content"]


async def test_auth_path_unaffected_by_kind_dispatch():
    ch = FeishuChannel(FeishuConfig(app_id="a", app_secret="s"))
    req = AuthRequest(request_id="r1", conversation_key="oc", command="rm x")
    ch._auth_requests["r1"] = req

    ch.handle_card_action(_ev({"action": "approve", "request_id": "r1"}))
    assert req.approved is True
    assert req.event.is_set()

    # 未知 request_id + 无 kind → 告警路径，不抛异常
    ch.handle_card_action(_ev({"action": "deny", "request_id": "ghost"}))

    # model_switch 但未接线 handler → 告警路径，不抛异常
    ch.handle_card_action(_ev({"kind": "model_switch", "stage": "providers"}))
