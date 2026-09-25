"""RecallEvalLogger 单元测试：评估集落盘的配对、标签与 fail-open 行为。"""

from __future__ import annotations

import json

from connectclaw.memory.recall_eval import RecallEvalLogger
from connectclaw.memory.store import MemoryStore
from connectclaw.memory.types import MemoryEntry, MemoryType, SearchResult


def _make_store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "mem.db"))


def _result(store: MemoryStore, content: str, score: float = 0.6) -> SearchResult:
    e = MemoryEntry(type=MemoryType.SEMANTIC, content=content, importance=0.5)
    store.add(e)
    return SearchResult(entry=e, score=score, detail_level="full",
                        similarity=0.55, bm25=0.1)


def _read_lines(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def test_start_finalize_writes_one_record(tmp_path):
    store = _make_store(tmp_path)
    path = tmp_path / "eval.jsonl"
    log = RecallEvalLogger(path)

    results = [_result(store, "用户的项目叫 ConnectClaw"),
               _result(store, "另一条没被用上的记忆")]
    log.start(query="项目叫什么", session_id="s1", conversation_key="c1",
              recalled_results=results, injected_ids={results[0].entry.id})
    log.finalize(results, "你的项目是 ConnectClaw", [results[0].entry.id])

    recs = _read_lines(path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["v"] == 1
    assert rec["query"] == "项目叫什么"
    assert rec["reply"] == "你的项目是 ConnectClaw"
    assert rec["used_ids"] == [results[0].entry.id]
    by_id = {c["id"]: c for c in rec["candidates"]}
    assert len(by_id) == 2
    assert by_id[results[0].entry.id]["newly_injected"] is True
    assert by_id[results[1].entry.id]["newly_injected"] is False
    assert by_id[results[0].entry.id]["similarity"] == 0.55


def test_persona_candidates_excluded(tmp_path):
    store = _make_store(tmp_path)
    path = tmp_path / "eval.jsonl"
    log = RecallEvalLogger(path)

    persona = _result(store, "称呼用户为老板", score=1.0)
    normal = _result(store, "普通记忆")
    results = [persona, normal]
    log.start(query="q", session_id="s", conversation_key="c",
              recalled_results=results, injected_ids=set())
    log.finalize(results, "reply", [])

    recs = _read_lines(path)
    assert len(recs) == 1
    assert [c["id"] for c in recs[0]["candidates"]] == [normal.entry.id]


def test_finalize_without_start_is_noop(tmp_path):
    store = _make_store(tmp_path)
    path = tmp_path / "eval.jsonl"
    log = RecallEvalLogger(path)

    results = [_result(store, "孤儿结果")]
    log.finalize(results, "reply", [results[0].entry.id])
    assert not path.exists()


def test_double_finalize_writes_once(tmp_path):
    store = _make_store(tmp_path)
    path = tmp_path / "eval.jsonl"
    log = RecallEvalLogger(path)

    results = [_result(store, "只确认一次")]
    log.start(query="q", session_id="s", conversation_key="c",
              recalled_results=results)
    log.finalize(results, "reply", [])
    log.finalize(results, "reply again", [])
    assert len(_read_lines(path)) == 1


def test_pending_cap_drops_oldest(tmp_path):
    store = _make_store(tmp_path)
    path = tmp_path / "eval.jsonl"
    log = RecallEvalLogger(path)

    batches = []
    for i in range(70):  # 超过 _MAX_PENDING=64
        results = [_result(store, f"记忆 {i}")]
        batches.append(results)
        log.start(query=f"q{i}", session_id="s", conversation_key="c",
                  recalled_results=results)
    # 最旧的一批已被挤出 → finalize 不落盘；最新的一批还在
    log.finalize(batches[0], "reply", [])
    log.finalize(batches[-1], "reply", [])
    recs = _read_lines(path)
    assert len(recs) == 1
    assert recs[0]["query"] == "q69"


def test_fail_open_on_unwritable_path(tmp_path):
    store = _make_store(tmp_path)
    # finalize 的目标路径是个目录 → 写入必炸，但绝不能抛异常
    log = RecallEvalLogger(tmp_path / "its_a_dir")
    (tmp_path / "its_a_dir").mkdir()

    results = [_result(store, "写不进去也不能崩")]
    log.start(query="q", session_id="s", conversation_key="c",
              recalled_results=results)
    log.finalize(results, "reply", [])  # 不应 raise

    # start 阶段：candidates 为空（全 persona）时也不崩、不落 pending
    persona = [_result(store, "只有 persona", score=1.0)]
    log.start(query="q", session_id="s", conversation_key="c",
              recalled_results=persona)
    log.finalize(persona, "reply", [])
