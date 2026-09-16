"""/model command tests — escape hatch flows with a fake CodingAgent.

The fake implements the small surface /model touches (list/save/apply/test +
wizard), so we verify command parsing, wizard round-trips and sub-command
routing without any model or network.
"""

from __future__ import annotations

from types import SimpleNamespace

from connectclaw.commands import handle, run_model_wizard_step
from connectclaw.model_registry import ModelProfile


class FakeAgent:
    def __init__(self) -> None:
        self.profiles: dict[str, ModelProfile] = {}
        self.active = ModelProfile(
            name="(active)", base_url="https://x", model_id="cur")
        self.wizard: tuple[str, dict] | None = None
        self._config = SimpleNamespace(proxy=SimpleNamespace(url=""))
        self.selected: str | None = None

    def entry_model_profile(self) -> ModelProfile:
        return self.active

    def list_model_profiles(self) -> list[ModelProfile]:
        return list(self.profiles.values())

    def get_model_profile(self, name: str) -> ModelProfile | None:
        return self.profiles.get(name)

    def save_model_profile(self, p: ModelProfile) -> str:
        if not p.name:
            base, i = p.model_id, 2
            while base in self.profiles:
                base = f"{p.model_id}-{i}"
                i += 1
            p.name = base
        self.profiles[p.name] = p
        return p.name

    def delete_model_profile(self, name: str) -> bool:
        return self.profiles.pop(name, None) is not None

    async def apply_model_profile(self, p: ModelProfile) -> str:
        self.selected = p.name
        self.active = p
        return f"✅ 已切换并持久化：**{p.name}**"

    async def test_model_profile(self, p: ModelProfile) -> tuple[bool, str]:
        if p.api_key:
            return True, "✅ 请求成功，模型可用。"
        return False, "请求失败：missing key"

    def set_wizard(self, conv: str, state: dict) -> None:
        self.wizard = (conv, state)

    def pop_wizard(self, conv: str) -> dict | None:
        if self.wizard and self.wizard[0] == conv:
            st, self.wizard = self.wizard, None
            return st[1]
        return None


def _fa() -> FakeAgent:
    return FakeAgent()


async def test_model_bare_lists_empty_and_active():
    fa = _fa()
    resp = await handle("/model", conversation_key="oc", agent=fa)
    assert "注册表为空" in resp and "当前激活" in resp


async def test_model_add_inline_saves():
    fa = _fa()
    resp = await handle(
        "/model add name=p1 base_url=https://api.deepseek.com model_id=deepseek-v4-flash api_key=sk-1",
        conversation_key="oc", agent=fa,
    )
    assert "已保存" in resp and "p1" in resp
    assert fa.profiles["p1"].base_url.endswith("deepseek.com")


async def test_model_add_missing_required_starts_wizard():
    fa = _fa()
    resp = await handle("/model add", conversation_key="oc", agent=fa)
    assert "base_url" in resp and fa.wizard is not None
    # wizard round-trip via chat replies
    r1 = await run_model_wizard_step(fa, fa.wizard[1], "not-a-url")
    assert "http" in r1  # rejected
    r2 = await run_model_wizard_step(fa, fa.wizard[1], "https://gw/v1")
    assert "model_id" in r2
    r3 = await run_model_wizard_step(fa, fa.wizard[1], "my-model")
    assert "api_key" in r3
    r4 = await run_model_wizard_step(fa, fa.wizard[1], "无")  # skip key
    assert "已保存" in r4
    assert fa.wizard is None
    assert any(p.model_id == "my-model" for p in fa.profiles.values())


async def test_model_set_activates_profile():
    fa = _fa()
    fa.save_model_profile(ModelProfile(name="p2", base_url="https://z", model_id="m2", api_key="k"))
    resp = await handle("/model set p2", conversation_key="oc", agent=fa)
    assert "已切换" in resp and fa.selected == "p2" and fa.active.model_id == "m2"


async def test_model_set_unknown_profile_errors():
    fa = _fa()
    resp = await handle("/model set ghost", conversation_key="oc", agent=fa)
    assert "没有名为" in resp


async def test_model_test_active_and_named():
    fa = _fa()
    resp1 = await handle("/model test", conversation_key="oc", agent=fa)
    assert "missing key" in resp1  # active has no key → failure path
    fa.save_model_profile(ModelProfile(name="good", base_url="https://z", model_id="m", api_key="sk"))
    resp2 = await handle("/model test good", conversation_key="oc", agent=fa)
    assert "请求成功" in resp2


async def test_model_edit_updates_field():
    fa = _fa()
    fa.save_model_profile(ModelProfile(name="e1", base_url="https://a", model_id="m", api_key="k"))
    resp = await handle("/model edit e1 model_id=m2 desc=更好", conversation_key="oc", agent=fa)
    assert "已更新" in resp and fa.profiles["e1"].model_id == "m2"


async def test_model_rm():
    fa = _fa()
    fa.save_model_profile(ModelProfile(name="gone", base_url="https://a", model_id="m"))
    resp = await handle("/model rm gone", conversation_key="oc", agent=fa)
    assert "已删除" in resp and "gone" not in fa.profiles


async def test_model_cancel_clears_wizard():
    fa = _fa()
    await handle("/model add", conversation_key="oc", agent=fa)
    assert fa.wizard is not None
    resp = await handle("/model cancel", conversation_key="oc", agent=fa)
    assert "取消" in resp and fa.wizard is None


async def test_model_help_lists_escape_hints():
    fa = _fa()
    for cmd in ("/model help", "/model -h", "/model unknownsub"):
        resp = await handle(cmd, conversation_key="oc", agent=fa)
        assert "/model" in resp and "test" in resp and "set" in resp
