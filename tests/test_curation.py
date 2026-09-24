"""做梦的"整理"相：同主题冲突/过时/环境可读事实的治理。

背景（2026-09-24 实测）：记忆库里「RAG已启用」（2026-07-27）被注入 84 次，而当天
写的更正只出现 1 次；同主题的旧说法与新说法以同等权威并存。用户定调：这类治理
放在**后台做梦**里做——读全量记忆 + 环境事实快照，让模型给出更正/遗忘/合并，再落库。
"""

from __future__ import annotations

import asyncio

import pytest

from connectclaw.coding.coding_agent import CodingAgent
from connectclaw.config import Config
from connectclaw.memory import consolidator as cons
from connectclaw.memory.consolidator import ConsolidationConfig, MemoryConsolidator, _stamp
from connectclaw.memory.store import MemoryStore
from connectclaw.memory.subsystem import MemoryConfig, MemorySubsystem
from connectclaw.memory.types import MemoryEntry, MemoryType
from connectclaw.provider.types import Model


STUB_MODEL = Model(id="stub-for-curation-test")


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    yield s
    s.close()


@pytest.fixture
def curator(store):
    return MemoryConsolidator(store, ConsolidationConfig())


def _add(store, content, **kw):
    e = MemoryEntry(type=kw.pop("type", MemoryType.SEMANTIC), content=content,
                    importance=kw.pop("importance", 0.9), **kw)
    store.add(e)
    return e


# ── 落库行为 ─────────────────────────────────────────────────

def test_curation_updates_corrected_memory(store, curator):
    e = _add(store, "当前模型路由走 USTC 代理")
    out = curator._apply_curation(
        {"update": [{"id": e.id, "content": "当前模型路由已直连 dots.ai，不走 USTC 代理",
                     "reason": "环境事实是直连"}]},
        [e],
    )
    assert out["curated"] == 1
    assert "直连 dots.ai" in store.get(e.id).content


def test_curation_forgets_stale_memory(store, curator):
    e = _add(store, "Vision模型配置为 qwen3.6-chat，RAG已启用")
    out = curator._apply_curation({"forget": [{"id": e.id, "reason": "环境可读且已停用"}]}, [e])
    assert out["purged"] == 1
    assert store.get(e.id) is None or (store.get(e.id).strength or 0) <= 0.05


def test_curation_merges_same_topic_memories(store, curator):
    a = _add(store, "项目用飞书做前端")
    b = _add(store, "项目的前端是飞书 IM")
    out = curator._apply_curation(
        {"merge_groups": [{"memory_ids": [a.id, b.id], "merged_content": "项目前端是飞书 IM"}]},
        [a, b],
    )
    assert out["curated"] == 1
    assert store.get(a.id).content == "项目前端是飞书 IM"
    assert store.get(b.id) is None


def test_curation_adds_new_semantic(store, curator):
    out = curator._apply_curation(
        {"new_semantic": [{"content": "用户偏好简短回复", "importance": 0.7, "reason": "多次确认"}]},
        [],
    )
    assert out["curated"] == 1
    assert any("简短回复" in e.content for e in store.list_all())


def test_curation_ignores_unknown_ids_and_empty_content(store, curator):
    e = _add(store, "甲")
    out = curator._apply_curation(
        {"update": [{"id": "不存在", "content": "注入"}, {"id": e.id, "content": "  "}],
         "forget": ["不存在", {"id": "也不存在"}]},
        [e],
    )
    assert out == {"curated": 0, "purged": 0}
    assert store.get(e.id).content == "甲"


def test_curation_noop_when_content_unchanged(store, curator):
    e = _add(store, "甲")
    out = curator._apply_curation({"update": [{"id": e.id, "content": "甲"}]}, [e])
    assert out["curated"] == 0


# ── 与 LLM 的一整圈（stub 掉模型调用）────────────────────────

def test_curate_applies_llm_output(store, curator, monkeypatch):
    e = _add(store, "RAG已启用")

    async def fake_llm(context, model, api_key=None):
        # 提示词里必须带上记忆与环境事实快照
        prompt = context.messages[0].content
        assert "<memories>" in prompt and e.id in prompt
        assert "<environment-facts>" in prompt and "RAG：disabled" in prompt
        return ('{"update": [], "forget": [{"id": "%s", "reason": "已停用"}], '
                '"merge_groups": [], "new_semantic": []}' % e.id)

    monkeypatch.setattr(cons, "_call_llm", fake_llm)
    out = asyncio.run(curator.curate(model=STUB_MODEL, env_facts="- RAG：disabled"))
    assert out["purged"] == 1


def test_curate_tolerates_unparseable_output(store, curator, monkeypatch):
    _add(store, "甲")

    async def bad_llm(context, model, api_key=None):
        return "这不是 JSON"

    monkeypatch.setattr(cons, "_call_llm", bad_llm)
    assert asyncio.run(
        curator.curate(model=STUB_MODEL, env_facts="")
    ) == {"curated": 0, "purged": 0}


def test_curate_skips_when_store_empty(curator):
    # 空库直接返回，连模型都不用叫
    assert asyncio.run(
        curator.curate(model=STUB_MODEL, env_facts="")
    ) == {"curated": 0, "purged": 0}


def test_curate_skips_without_model(curator, store):
    """没有可用模型时跳过整理相——"无 LLM 的做梦"仍要能跑完。"""
    _add(store, "甲")
    assert asyncio.run(curator.curate(model=None, env_facts="")) == {"curated": 0, "purged": 0}


# ── 环境事实快照 ─────────────────────────────────────────────

def test_stamp_includes_date_and_strength(store):
    e = _add(store, "甲", strength=0.31, created_at=1_785_000_000.0)
    stamp = _stamp(e)
    assert stamp.startswith("[") and stamp.endswith("]")
    assert "0.31" in stamp


def test_environment_facts_snapshot_looks_sane():
    """采集器要能报出"能直接从环境读出来"的那几件事。"""
    agent = CodingAgent(Config())
    facts = agent.environment_facts()
    assert "当前激活模型" in facts
    assert "RAG" in facts
    assert "工作目录" in facts
    assert "采集时间" in facts


def test_subsystem_env_facts_provider_never_breaks_dream(tmp_path):
    mem = MemorySubsystem(MemoryConfig(enabled=True, db_path=str(tmp_path / "m.db")))
    assert mem._env_facts() == ""                      # 未接线 → 空
    mem.set_env_facts_provider(lambda: "环境快照")

    def boom() -> str:
        raise RuntimeError("探测失败")

    mem.set_env_facts_provider(boom)
    assert mem._env_facts() == "", "环境探测失败必须降级为空串，不能拖垮做梦"


def test_curation_prompt_protects_technical_knowledge():
    """API/协议/工具踩坑这类"交学费学来的"知识不是环境可读的，要留。

    2026-09-25 实跑一次真实 /dream 后发现的：它把「dots3-note-prev 返回
    reasoning_content 字段」当成"能读出来"的删了——那条其实是踩出来的 API 知识。
    """
    from connectclaw.memory.prompts import CURATION_PROMPT

    assert "KEEP TECHNICAL KNOWLEDGE" in CURATION_PROMPT
    assert "correct it instead of deleting it" in CURATION_PROMPT
