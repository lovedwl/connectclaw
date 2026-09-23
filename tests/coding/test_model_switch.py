"""/model set — a live switch must move the API KEY with the endpoint.

Regression: `apply_model_profile` swapped the Model object but left the
in-memory `[llm]` config alone, while every request resolves its key from
`config.llm.api_key` (harness, agents tool, memory extraction, dream). So the
NEW endpoint received the PREVIOUS provider's key, which gateways reject with
403 — dots.ai answers exactly `governance.dots_platform_key_not_allowed`. The
new model "worked" only after a restart re-read config.toml.

Also covers the 404 → /v1 diagnosis that `/model test` now performs.
"""

from __future__ import annotations

from types import SimpleNamespace

from connectclaw.coding import coding_agent as ca
from connectclaw.coding.coding_agent import CodingAgent
from connectclaw.config import Config
from connectclaw.model_registry import ModelProfile


class _RecordingStore:
    """Stands in for ModelsStore so tests never touch ~/.connectclaw."""

    saved: list[ModelProfile] = []

    def __init__(self, *a, **kw):
        pass

    def set_active(self, p: ModelProfile) -> None:
        _RecordingStore.saved.append(p)


def _agent(monkeypatch, *, key: str = "sk-old-gateway", base: str = "https://old.example.com/v1"):
    monkeypatch.setattr(ca, "ModelsStore", _RecordingStore)
    _RecordingStore.saved = []
    cfg = Config()
    cfg.llm.api_key = key
    cfg.llm.base_url = base
    cfg.llm.model_id = "old-model"
    return CodingAgent(cfg), cfg


async def test_apply_profile_moves_key_and_endpoint_together(monkeypatch):
    ag, cfg = _agent(monkeypatch)
    p = ModelProfile(
        name="dots3-note-prev",
        base_url="https://note3-prev-api.askdiandian.com/v1",
        model_id="dots3-note-prev",
        api_key="ak-new-key",
    )

    msg = await ag.apply_model_profile(p)

    # the live config — where every request reads its key from
    assert cfg.llm.api_key == "ak-new-key"
    assert cfg.llm.base_url == p.base_url
    assert cfg.llm.model_id == "dots3-note-prev"
    # the model object handed to the harness
    assert ag._model.base_url == p.base_url and ag._model.id == "dots3-note-prev"
    # sub-agents resolve the same key at call time
    assert ag._agents_tool._resolve_api_key() == "ak-new-key"
    # the reply shows a fingerprint, not the secret (short keys mask harder)
    assert "ak-new-key" not in msg and "ak_new" not in msg
    assert "ak-…" in msg


async def test_keyless_profile_inherits_key_only_on_the_same_host(monkeypatch):
    """同一个网关换 model_id 可以不带 key；跨网关时必须清掉旧 key。"""
    ag, cfg = _agent(monkeypatch)
    same_host = ModelProfile(name="same", base_url="https://old.example.com/v1",
                             model_id="other-model", api_key="")
    await ag.apply_model_profile(same_host)
    assert cfg.llm.api_key == "sk-old-gateway"  # inherited: same gateway

    ag2, cfg2 = _agent(monkeypatch)
    other_host = ModelProfile(name="other", base_url="https://dots.example.com/v1",
                              model_id="m", api_key="")
    msg = await ag2.apply_model_profile(other_host)
    assert cfg2.llm.api_key == ""  # never send another provider's key
    assert "没有 api_key" in msg


async def test_resolved_env_key_reaches_the_live_config(monkeypatch):
    monkeypatch.setenv("MY_DOTS_KEY", "ak-from-env")
    ag, cfg = _agent(monkeypatch)
    p = ModelProfile(name="env", base_url="https://dots.example.com/v1",
                     model_id="m", api_key="$MY_DOTS_KEY")
    await ag.apply_model_profile(p)
    assert cfg.llm.api_key == "ak-from-env"


# ── /model test: the missing-/v1 diagnosis ─────────────────────


def _stream_returning(script: dict[str, str]):
    """Fake stream_simple: base_url → "done" | error message."""
    calls: list[str] = []

    async def fake(model, context, **kwargs):
        calls.append(model.base_url)
        outcome = script.get(model.base_url, "404 page not found")
        if outcome == "done":
            yield SimpleNamespace(type="done", error_message=None)
        else:
            yield SimpleNamespace(type="error", error_message=outcome)

    return fake, calls


async def test_model_test_retries_v1_when_bare_endpoint_404s(monkeypatch):
    fake, calls = _stream_returning({
        "https://note3-prev-api.askdiandian.com/v1": "done",
    })
    monkeypatch.setattr(ca, "stream_simple", fake)
    ag, _ = _agent(monkeypatch)
    p = ModelProfile(name="dots3-note-prev",
                     base_url="https://note3-prev-api.askdiandian.com",
                     model_id="dots3-note-prev", api_key="ak-1")

    ok, detail = await ag.test_model_profile(p)

    assert calls == ["https://note3-prev-api.askdiandian.com",
                     "https://note3-prev-api.askdiandian.com/v1"]
    # not "ok": the profile AS SAVED still 404s — say what to fix instead
    assert ok is False
    assert "缺 `/v1` 前缀" in detail
    assert "/model edit dots3-note-prev base_url=https://note3-prev-api.askdiandian.com/v1" in detail


async def test_model_test_reports_v1_404_without_guessing_further(monkeypatch):
    fake, calls = _stream_returning({})
    monkeypatch.setattr(ca, "stream_simple", fake)
    ag, _ = _agent(monkeypatch)
    p = ModelProfile(name="dots", base_url="https://dots.example.com/v1",
                     model_id="m", api_key="ak-1")

    ok, detail = await ag.test_model_profile(p)

    assert calls == ["https://dots.example.com/v1"]  # already /v1: no retry
    assert ok is False and "404" in detail and "/v1" in detail


async def test_model_test_does_not_probe_on_non_404(monkeypatch):
    fake, calls = _stream_returning({"https://dots.example.com": "Error code: 401 - bad key"})
    monkeypatch.setattr(ca, "stream_simple", fake)
    ag, _ = _agent(monkeypatch)
    p = ModelProfile(name="dots", base_url="https://dots.example.com",
                     model_id="m", api_key="ak-1")

    ok, detail = await ag.test_model_profile(p)

    assert calls == ["https://dots.example.com"]  # a 401 is not a path problem
    assert ok is False and "401" in detail


async def test_model_test_uses_the_profiles_own_key(monkeypatch):
    """test and set must agree on the key, or a profile tests green and runs 403."""
    keys: list[str | None] = []

    async def fake(model, context, **kwargs):
        keys.append(kwargs.get("api_key"))
        yield SimpleNamespace(type="done", error_message=None)

    monkeypatch.setattr(ca, "stream_simple", fake)
    ag, _ = _agent(monkeypatch)
    p = ModelProfile(name="dots", base_url="https://dots.example.com/v1",
                     model_id="m", api_key="ak-profile")

    ok, _ = await ag.test_model_profile(p)

    assert ok is True and keys == ["ak-profile"]


async def test_active_config_404_gets_a_config_hint_not_an_edit_hint(monkeypatch):
    """激活配置的 profile 名是 "(active)"，不该叫你 `/model edit (active)`。"""
    fake, calls = _stream_returning({"https://old.example.com/v1": "done"})
    monkeypatch.setattr(ca, "stream_simple", fake)
    ag, _ = _agent(monkeypatch, base="https://old.example.com")  # no /v1 in [llm]

    ok, detail = await ag.test_model_profile(ag.entry_model_profile())

    assert calls == ["https://old.example.com", "https://old.example.com/v1"]
    assert ok is False
    assert "config.toml" in detail and "/restart" in detail
    assert "/model edit (active)" not in detail
