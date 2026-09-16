"""ModelsStore tests — profile registry CRUD + surgical [llm] activation.

Pins two escape-critical properties: the registry lives in its OWN file (so
config.toml comments survive), and set_active updates ONLY the [llm] block.
"""

from __future__ import annotations

from connectclaw.model_registry import ModelProfile, ModelsStore


def _store(tmp_path):
    return ModelsStore(
        registry_path=str(tmp_path / "models.toml"),
        config_path=str(tmp_path / "config.toml"),
    )


def test_save_auto_names_and_dedups(tmp_path):
    store = _store(tmp_path)
    n1 = store.save(ModelProfile(name="", base_url="https://a", model_id="m1", api_key="k1"))
    assert n1 == "m1"
    n2 = store.save(ModelProfile(name="", base_url="https://a", model_id="m1", api_key="k1"))
    assert n2 == "m1-2"
    assert len(store.list()) == 2


def test_crud_roundtrip_persists(tmp_path):
    store = _store(tmp_path)
    store.save(ModelProfile(
        name="p1", base_url="https://a/", model_id="m1", api_key="sk-x",
        context_window=131072, max_tokens=16384, reasoning=False, desc="测试",
    ))
    # fresh store instance reloads from disk
    store2 = _store(tmp_path)
    p = store2.get("p1")
    assert p is not None
    assert p.base_url == "https://a/" and p.model_id == "m1" and p.api_key == "sk-x"
    assert p.context_window == 131072 and p.max_tokens == 16384
    assert p.reasoning is False and p.desc == "测试"
    assert store2.delete("p1") is True
    assert _store(tmp_path).get("p1") is None


def test_unique_names_preserved(tmp_path):
    store = _store(tmp_path)
    store.save(ModelProfile(name="a", base_url="https://x", model_id="m"))
    store.save(ModelProfile(name="b", base_url="https://y", model_id="m"))
    assert {p.name for p in store.list()} == {"a", "b"}


# ── activation: [llm] surgical replace ─────────────────────────


_FIXTURE = """\
# 顶部注释
# [llm]
# api_key = "commented-old"

[llm]
api_key = "old-key"
base_url = "https://old/base"
model_id = "old-model"

[feishu]
app_id = "preserve-me"
"""
_FIXTURE_PATH_UNUSED = True


def test_set_active_replaces_only_llm_and_keeps_comments(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(_FIXTURE, encoding="utf-8")
    store = ModelsStore(registry_path=str(tmp_path / "models.toml"), config_path=str(cfg))

    store.set_active(ModelProfile(
        name="p2", base_url="https://new/base", model_id="new-model",
        api_key="sk-new", context_window=262144, max_tokens=32768, reasoning=True,
    ))

    text = cfg.read_text(encoding="utf-8")
    # [llm] block updated
    assert 'api_key = "sk-new"' in text
    assert 'base_url = "https://new/base"' in text
    assert 'model_id = "new-model"' in text
    assert "262144" in text and "32768" in text
    # comments + other sections untouched
    assert "# [llm]" in text and 'api_key = "commented-old"' in text
    assert 'app_id = "preserve-me"' in text
    assert "old-key" not in text


def test_set_active_skips_empty_api_key(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[llm]\napi_key = \"x\"\nbase_url = \"a\"\nmodel_id = \"m\"\n", encoding="utf-8")
    store = ModelsStore(registry_path=str(tmp_path / "models.toml"), config_path=str(cfg))
    store.set_active(ModelProfile(name="p", base_url="https://z", model_id="m2", api_key=""))
    text = cfg.read_text(encoding="utf-8")
    assert "api_key = \"\"" not in text
    assert 'model_id = "m2"' in text


def test_resolved_api_key_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_ESC_KEY", "sk-secret")
    p = ModelProfile(name="p", base_url="https://a", model_id="m", api_key="$MY_ESC_KEY")
    assert p.resolved_api_key() == "sk-secret"
