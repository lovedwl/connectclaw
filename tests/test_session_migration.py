"""旧会话迁移脚本（scripts/migrate_sessions_to_structured.py）的测试。

旧格式是"注入渲染成文本拼进用户消息"，已一次性迁移成结构化 `context` 条目；
迁移会**改写会话语料**，所以它自己也值得被钉住：结构正确、幂等、父链不断。
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "migrate_sessions_to_structured",
    Path(__file__).resolve().parents[1] / "scripts" / "migrate_sessions_to_structured.py",
)
migrate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrate)  # type: ignore[union-attr]


LEGACY_INJECTED = (
    "<remembered-context>\n(Things you know from past interactions)\n"
    "- [2026-07-27 · 1.00] 用户要求全程中文\n"
    "- [2026-09-24 · 0.99] 当前模型路由已直连 dots.ai\n"
    "</remembered-context>\n\n"
    "## 可运行的 agents(用 `agents(action=\"run\", agent=\"<名>\")` 调用)\n"
    "- search — 网络搜索专家\n\n"
    "帮我查查 jev"
)


def _fixture(path: Path) -> None:
    rows = [
        {"type": "session", "version": 3, "id": "t", "created_at": "2026-09-01T00:00:00Z", "cwd": "/"},
        {"type": "message", "id": "u1", "parent_id": None, "timestamp": "2026-09-01T00:00:01Z",
         "message": {"role": "user", "content": LEGACY_INJECTED}},
        {"type": "message", "id": "a1", "parent_id": "u1", "timestamp": "2026-09-01T00:00:02Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "好的"}]}},
        {"type": "message", "id": "u2", "parent_id": "a1", "timestamp": "2026-09-01T00:00:03Z",
         "message": {"role": "user", "content": [
             {"type": "text", "text": LEGACY_INJECTED.replace("帮我查查 jev", "再看一眼")},
             {"type": "image_ref", "id": "img1", "path": "/tmp/x.png",
              "mime_type": "image/png", "size": 10},
         ]}},
        {"type": "compaction", "id": "cp1", "parent_id": "u2", "timestamp": "2026-09-01T01:00:00Z",
         "summary": "摘要", "first_kept_entry_id": "u2", "tokens_before": 10,
         "merged_context": "<remembered-context>\n- 远古记忆\n</remembered-context>"},
    ]
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture()
def legacy_session(tmp_path):
    path = tmp_path / "s.jsonl"
    _fixture(path)
    return path


def test_migrates_message_into_context_entry(legacy_session):
    rows, stats = migrate.migrate_file(str(legacy_session), set())
    assert stats["contexts_added"] == 2
    assert stats["compactions_annotated"] == 1

    by_id = {r.get("id"): r for r in rows}
    ctx = [r for r in rows if r.get("type") == "context"]
    assert len(ctx) == 2
    # 注入变成了结构化 op
    kinds = [o["op"] for o in ctx[0]["ops"]]
    assert kinds.count("memory_add") == 2 and "catalog_set" in kinds
    # 用户消息变干净，父链改接到新条目
    assert by_id["u1"]["message"]["content"] == "帮我查查 jev"
    assert by_id["u1"]["parent_id"] == ctx[0]["id"]
    assert ctx[0]["parent_id"] is None
    # 列表形状的 content：只换文本块，图片块原样保留
    u2 = by_id["u2"]["message"]["content"]
    assert u2[0]["text"] == "再看一眼"
    assert u2[1]["type"] == "image_ref"
    # 压缩条目补上了结构化快照（后续折叠不必再解析文本）
    assert by_id["cp1"]["context_state"]["op"] == "state_snapshot"
    assert "远古记忆" in str(by_id["cp1"]["context_state"]["memory"])


def test_result_passes_validation(legacy_session):
    rows, _ = migrate.migrate_file(str(legacy_session), set())
    assert migrate.validate(str(legacy_session), rows) == []


def _write(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_second_run_is_noop(legacy_session):
    rows, _ = migrate.migrate_file(str(legacy_session), set())
    _write(legacy_session, rows)                      # 落盘（等价于 --apply）
    again, stats2 = migrate.migrate_file(str(legacy_session), set())
    assert stats2 == {"contexts_added": 0, "compactions_annotated": 0, "only_injection": 0}
    assert again == rows, "再跑一次不该有任何变化"


def test_validate_flags_unmigrated_user_message(legacy_session):
    rows = [json.loads(l) for l in legacy_session.read_text(encoding="utf-8").splitlines() if l.strip()]
    problems = migrate.validate(str(legacy_session), rows)
    assert any("仍含注入文本" in p for p in problems)


def test_toolresult_containing_injection_text_is_not_flagged(legacy_session):
    """工具读过会话文件、把注入文本当内容返回 —— 这是正常数据，不该被判为未迁移。"""
    rows, _ = migrate.migrate_file(str(legacy_session), set())   # 先迁移
    rows.append({
        "type": "message", "id": "t1", "parent_id": "u2", "timestamp": "2026-09-01T02:00:00Z",
        "message": {"role": "toolResult", "tool_call_id": "c1", "tool_name": "read",
                    "content": [{"type": "text", "text": LEGACY_INJECTED}], "is_error": False},
    })
    _write(legacy_session, rows)
    assert migrate.validate(str(legacy_session), rows) == []
