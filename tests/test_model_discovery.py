"""Tests for /model auto-discovery (2026-10): model_discovery + card stages
+ the discovery-aware wizard flows. A real local HTTP server stands in for
the OpenAI-compatible gateway."""
from __future__ import annotations

import http.server
import json as _json
import threading

import pytest

from connectclaw import model_discovery
from connectclaw.model_discovery import DiscoveryError, fmt_ctx, match_existing, parse_entry
from connectclaw.model_registry import ModelProfile


# ── unit: parsing ──────────────────────────────────────────────


def test_parse_entry_variants():
    e = parse_entry({"id": "m1", "max_input_tokens": 131072, "max_output_tokens": 4096})
    assert e == {"id": "m1", "max_input": 131072, "max_output": 4096}
    e = parse_entry({"id": "m2"})
    assert e == {"id": "m2", "max_input": None, "max_output": None}
    e = parse_entry({"id": 5, "max_input": "262144"})  # int id, string limit
    assert e == {"id": "5", "max_input": 262144, "max_output": None}


def test_fmt_ctx():
    assert fmt_ctx(None) == "?"
    assert fmt_ctx(0) == "?"
    assert fmt_ctx(262144) == "256k"
    assert fmt_ctx(1048576) == "1M"
    assert fmt_ctx(500) == "500"


def test_extract_limit_variants():
    f = model_discovery._extract_limit
    assert f("This model's maximum context length is 262144 tokens. "
             "However, your messages resulted in 1600016 tokens.") == 262144
    assert f("maximum context length is 128,000 tokens") == 128000
    assert f("prompt is too long: 204800 tokens > 128000 maximum") == 128000
    assert f("no numbers here") is None


def test_match_existing_host_based():
    cur = ModelProfile(name="p", base_url="https://gw.example.com/v1", model_id="m1")
    other_host = ModelProfile(name="q", base_url="https://gw2.example.com", model_id="m1")
    assert match_existing([cur], "https://gw.example.com/", "m1") is cur   # /v1 suffix ignored
    assert match_existing([cur], "https://gw.example.com", "m1").name == "p"
    assert match_existing([other_host], "https://gw.example.com", "m1") is None
    assert match_existing([cur], "https://gw.example.com", "m2") is None


# ── fake gateway ───────────────────────────────────────────────


class _GatewayHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/models":
            self.send_response(404)
            self.end_headers()
        elif self.path.endswith("/v1/models"):
            body = _json.dumps({"data": [
                {"id": "m1", "max_input_tokens": 131072, "max_output_tokens": 4096},
                {"id": "m2"},
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        self.rfile.read(ln)
        body = _json.dumps({"error": {"message": (
            "This model's maximum context length is 262,144 tokens. "
            "However, your messages resulted in 999999 tokens.")}}).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def fake_gateway():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _GatewayHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.server_close()


async def test_fetch_model_list_v1_fallback(fake_gateway):
    # base WITHOUT /v1: bare /models 404s, the /v1 candidate must win
    entries, endpoint = await model_discovery.fetch_model_list(fake_gateway, "sk-test")
    assert [e["id"] for e in entries] == ["m1", "m2"]
    assert endpoint == fake_gateway + "/v1"
    m1 = entries[0]
    assert m1["max_input"] == 131072 and m1["max_output"] == 4096


async def test_fetch_model_list_unreachable():
    with pytest.raises(DiscoveryError):
        await model_discovery.fetch_model_list("http://127.0.0.1:1/v1", "", timeout=2)


async def test_probe_parses_limit_from_400(fake_gateway):
    limit, note = await model_discovery.probe_context_window(
        fake_gateway + "/v1", "sk-test", "m1")
    assert limit == 262144
    assert "400" in note


# ── selection parser ───────────────────────────────────────────


def test_parse_selection():
    from connectclaw.commands import _parse_selection as f
    assert f("1,3", 5) == [1, 3]
    assert f("1~5", 5) == [1, 2, 3, 4, 5]
    assert f("3-1", 5) == [1, 2, 3]
    assert f("all", 3) == [1, 2, 3]
    assert f("全部", 2) == [1, 2]
    assert f("1, 3", 5) == [1, 3]
    assert f("abc", 5) is None
    assert f("9", 5) == []


# ── wizard flows (add + sync) ──────────────────────────────────


class FakeAgent:
    def __init__(self):
        self._store: dict[str, ModelProfile] = {}
        self._wizard: dict[str, dict] = {}
        self._config = None

    def list_model_profiles(self):
        return list(self._store.values())

    def get_model_profile(self, name):
        return self._store.get(name)

    def save_model_profile(self, p: ModelProfile) -> str:
        name = (p.name or "").strip() or p.model_id
        i = 2
        while name in self._store:
            name = f"{p.model_id}-{i}"
            i += 1
        p.name = name
        self._store[name] = p
        return name

    def set_wizard(self, k, state):
        self._wizard[k] = state

    def pop_wizard(self, k):
        return self._wizard.pop(k, None)

    def delete_model_profile(self, name):
        return self._store.pop(name, None) is not None


async def test_wizard_add_auto_discovery(monkeypatch):
    import connectclaw.commands as cmd

    async def fake_fetch(base, key, proxy=None, timeout=None):
        return ([{"id": "m1", "max_input": 131072, "max_output": 4096},
                 {"id": "m2", "max_input": None, "max_output": None}],
                base + "/v1")

    monkeypatch.setattr(model_discovery, "fetch_model_list", fake_fetch)
    ag = FakeAgent()
    state = {"fields": {}, "step": 0, "_conv": "oc", "mode": "add"}

    r1 = await cmd.run_model_wizard_step(ag, state, "gw.example.com")
    assert "api_key" in r1
    assert state["fields"]["base_url"] == "https://gw.example.com"

    r2 = await cmd.run_model_wizard_step(ag, state, "sk-123")
    assert "m1" in r2 and "128k" in r2
    assert state["mode"] == "add_pick"

    r3 = await cmd.run_model_wizard_step(ag, state, "1")
    assert "已保存" in r3 and "128k" in r3
    assert ag._store["m1"].context_window == 131072
    assert ag._store["m1"].max_tokens == 4096
    assert ag._store["m1"].api_key == "sk-123"
    assert "oc" not in ag._wizard


async def test_wizard_add_discovery_failure_falls_back(monkeypatch):
    import connectclaw.commands as cmd

    async def broken(base, key, proxy=None, timeout=None):
        raise DiscoveryError("connection refused")

    monkeypatch.setattr(model_discovery, "fetch_model_list", broken)
    ag = FakeAgent()
    state = {"fields": {}, "step": 0, "_conv": "oc", "mode": "add"}
    await cmd.run_model_wizard_step(ag, state, "gw.example.com")
    r2 = await cmd.run_model_wizard_step(ag, state, "无")
    assert "自动发现失败" in r2 and "model_id" in r2
    r3 = await cmd.run_model_wizard_step(ag, state, "my-model")
    assert "已保存" in r3
    assert ag._store["my-model"].context_window == 65536  # default


async def test_wizard_add_pick_by_name_and_cancel():
    import connectclaw.commands as cmd

    ag = FakeAgent()
    state = {"fields": {"base_url": "https://g/", "api_key": "k"},
             "step": 2, "_conv": "oc", "mode": "add_pick",
             "items": [{"id": "m1", "ctx": None, "mt": None}]}
    r = await cmd.run_model_wizard_step(ag, state, "m1")  # exact name pick
    assert "已保存" in r
    # cancel path
    state2 = {"fields": {}, "step": 0, "_conv": "oc", "mode": "add"}
    ag.set_wizard("oc", state2)
    r2 = await cmd.run_model_wizard_step(ag, state2, "取消")
    assert "已取消" in r2
    assert "oc" not in ag._wizard


async def test_wizard_sync_pick_add_update_skip():
    import connectclaw.commands as cmd

    ag = FakeAgent()
    cur = ModelProfile(name="example-qwen3.8-reasoner",
                       base_url="https://api.example.com/",
                       model_id="qwen3.8-reasoner", api_key="k")
    ag._store[cur.name] = cur
    state = {"mode": "sync_pick", "provider": "api.example.com",
             "base_url": "https://api.example.com/", "raw_key": "k",
             "endpoint": "e", "_conv": "oc",
             "items": [
                 {"id": "qwen3.8-reasoner", "ctx": 262144, "mt": 8192,
                  "exists": "example-qwen3.8-reasoner"},
                 {"id": "new-model", "ctx": None, "mt": None, "exists": None},
                 {"id": "stale", "ctx": 32768, "mt": None, "exists": "gone"},
             ],
             "existing": {"qwen3.8-reasoner": "example-qwen3.8-reasoner",
                          "new-model": None, "stale": "gone"},
             "fields": {}}
    r = await cmd.run_model_wizard_step(ag, state, "1,2,3")
    assert "更新" in r and "新增" in r
    assert ag._store["example-qwen3.8-reasoner"].context_window == 262144
    assert "new-model" in ag._store
    assert ag._store["new-model"].context_window == 65536  # default
    # bad selection syntax
    ag.set_wizard("oc", state)
    r2 = await cmd.run_model_wizard_step(ag, state, "hello")
    assert "回复格式" in r2


# ── card stages ────────────────────────────────────────────────


def _card_texts(card: dict) -> str:
    out = []

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("content"), str):
                out.append(x["content"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(card)
    return "\n".join(out)


def _card_values(card: dict) -> list[dict]:
    out = []

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("value"), dict) and x.get("tag") == "button":
                out.append(x["value"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(card)
    return out


def test_discovery_card_buttons_and_nonce():
    from connectclaw import model_card

    payload = {"provider": "P", "base_url": "https://x/", "raw_key": "k",
               "resolved_key": "k", "endpoint": "e",
               "items": [{"id": "a", "ctx": None, "mt": None, "exists": None},
                         {"id": "b", "ctx": 131072, "mt": 4096, "exists": "pb"}]}
    card = model_card.discovery_card(payload)
    vals = _card_values(card)
    stages = {v.get("stage") for v in vals}
    assert {"add_model", "update_model", "add_all", "models", "probe_unknown"} <= stages
    assert all("n" in v for v in vals)  # dedup nonce on every button
    texts = _card_texts(card)
    assert "128k" in texts and "已有" in texts


class FakeChannel:
    def __init__(self):
        self.updates: list[tuple[str, dict]] = []

    async def update_card(self, message_id, card):
        self.updates.append((message_id, card))
        return True


async def test_card_add_all_stage():
    from connectclaw import model_card

    ag = FakeAgent()
    pb = ModelProfile(name="pb", base_url="https://x/", model_id="b", api_key="k")
    ag._store["pb"] = pb
    ch = FakeChannel()
    ag._channel = ch
    payload = {"provider": "P", "base_url": "https://x/", "raw_key": "k",
               "resolved_key": "k", "endpoint": "e",
               "items": [{"id": "a", "ctx": 131072, "mt": 4096, "exists": None},
                         {"id": "b", "ctx": 65536, "mt": None, "exists": "pb"},
                         {"id": "c", "ctx": None, "mt": None, "exists": None}]}
    model_card._cache_put("m1", payload)
    await model_card.on_model_card_action(ag, {"stage": "add_all"}, "chat", "m1")

    assert "a" in ag._store and ag._store["a"].context_window == 131072
    assert "c" in ag._store and ag._store["c"].context_window == 65536
    assert ag._store["pb"].context_window == 65536
    text = _card_texts(ch.updates[-1][1])
    assert "批量同步完成" in text
    assert "新增 2" in text and "更新 1" in text and "跳过 0" in text


async def test_card_add_model_and_stale_cache():
    from connectclaw import model_card

    ag = FakeAgent()
    ch = FakeChannel()
    ag._channel = ch
    payload = {"provider": "P", "base_url": "https://x/", "raw_key": "k",
               "resolved_key": "k", "endpoint": "e",
               "items": [{"id": "a", "ctx": None, "mt": None, "exists": "exists-p"}]}
    model_card._cache_put("m1", payload)
    # stale cache → friendly message
    await model_card.on_model_card_action(ag, {"stage": "add_model", "id": "a"},
                                          "chat", "no-such-message")
    assert "已过期" in _card_texts(ch.updates[-1][1])
    # existing model → pointed to update
    await model_card.on_model_card_action(ag, {"stage": "add_model", "id": "a"},
                                          "chat", "m1")
    assert "已有 profile" in _card_texts(ch.updates[-1][1])
    # clean add
    payload["items"][0]["exists"] = None
    model_card._cache_put("m1", payload)
    await model_card.on_model_card_action(ag, {"stage": "add_model", "id": "a"},
                                          "chat", "m1")
    assert "a" in ag._store
    assert "已添加" in _card_texts(ch.updates[-1][1])


async def test_card_probe_unknown_fills_ctx(monkeypatch):
    from connectclaw import model_card

    async def fake_probe(base, key, model_id, proxy=None, timeout=None):
        return (999999, "HTTP 400（small 级探测命中）")

    monkeypatch.setattr(model_discovery, "probe_context_window", fake_probe)
    ag = FakeAgent()
    ch = FakeChannel()
    ag._channel = ch
    payload = {"provider": "P", "base_url": "https://x/", "raw_key": "k",
               "resolved_key": "k", "endpoint": "e",
               "items": [{"id": "a", "ctx": None, "mt": None, "exists": None},
                         {"id": "b", "ctx": 131072, "mt": 4096, "exists": None}]}
    model_card._cache_put("m1", payload)
    await model_card.on_model_card_action(ag, {"stage": "probe_unknown"}, "chat", "m1")
    assert payload["items"][0]["ctx"] == 999999
    text = _card_texts(ch.updates[-1][1])
    assert "探测完成" in text and "977k" in text


async def test_card_gone_detection_and_delete(monkeypatch):
    """网关下架检测：本地有但 /models 没有 → 红区 + 🗑；点击真删。"""
    from connectclaw import model_card

    ag = FakeAgent()
    ag._store["dead-profile"] = ModelProfile(
        name="dead-profile", base_url="https://x/", model_id="dead-model",
        api_key="k", provider="P")
    ch = FakeChannel()
    ag._channel = ch

    async def fake_fetch(base, key, proxy=None, timeout=None):
        return ([{"id": "alive", "max_input": 131072, "max_output": 4096}], "e")

    monkeypatch.setattr(model_card.model_discovery, "fetch_model_list", fake_fetch)
    await model_card.on_model_card_action(
        ag, {"stage": "discover", "provider": "P"}, "chat", "m1")
    card = ch.updates[-1][1]
    assert "已下架" in _card_texts(card)
    assert any(v.get("stage") == "delete_model" for v in _card_values(card))
    # 存活的 alive 不能被误标：行内按钮应是 ➕ 添加
    assert any(v.get("stage") == "add_model" and v.get("id") == "alive"
               for v in _card_values(card))

    await model_card.on_model_card_action(
        ag, {"stage": "delete_model", "name": "dead-profile"}, "chat", "m1")
    assert "dead-profile" not in ag._store
    assert "已删除" in _card_texts(ch.updates[-1][1])
    # 删完后卡片里不再有下架区
    assert "已下架" not in _card_texts(ch.updates[-1][1])


async def test_card_delete_guardrails():
    """🗑 只删清单里标过的名字；正在使用的模型拒删。"""
    from connectclaw import model_card

    ag = FakeAgent()
    ag._store["cur"] = ModelProfile(name="cur", base_url="https://x/",
                                    model_id="gone1", api_key="k", provider="P")
    ch = FakeChannel()
    ag._channel = ch
    payload = {"provider": "P", "base_url": "https://x/", "raw_key": "k",
               "resolved_key": "k", "endpoint": "e",
               "items": [{"id": "gone1", "ctx": None, "mt": None,
                          "exists": "cur", "gone": True}]}
    model_card._cache_put("m2", payload)

    # 构造 name 不在下架清单 → 拒绝
    await model_card.on_model_card_action(
        ag, {"stage": "delete_model", "name": "hacker"}, "chat", "m2")
    assert "不在已下架清单" in _card_texts(ch.updates[-1][1])

    # 正在使用 → 拒删
    ag.active_model_name = "cur"
    await model_card.on_model_card_action(
        ag, {"stage": "delete_model", "name": "cur"}, "chat", "m2")
    assert "正在使用中" in _card_texts(ch.updates[-1][1])
    assert "cur" in ag._store
