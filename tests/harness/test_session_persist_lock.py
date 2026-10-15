"""2026-10 (H5) regression tests — session persistence serialization.

Covers:
- concurrent append_message calls produce a linear parent chain (not siblings)
- concurrent appends don't interleave JSONL writes (one JSON object per line)
"""

from __future__ import annotations

import asyncio
import json

from connectclaw.agent.harness.session import JsonlSessionStorage


async def test_concurrent_appends_form_linear_chain(tmp_path):
    path = str(tmp_path / "s.jsonl")
    storage = await JsonlSessionStorage.create(path, cwd="/tmp")

    # 同一同步批次内连续创建的落盘任务（H5 场景）：此前全部读到同一个
    # 尚未更新的 leaf，10 条消息全部挂在同一父节点上。
    ids = await asyncio.gather(*[
        storage.append_message({"role": "user", "content": f"m{i}"})
        for i in range(10)
    ])

    entries = storage.entries
    assert len(entries) == 10
    assert [e.id for e in entries] == ids
    # 线性链：每条的 parent 恰好是前一条
    parents = [e.parent_id for e in entries]
    assert parents == [None] + [e.id for e in entries[:-1]]


async def test_concurrent_appends_do_not_corrupt_jsonl(tmp_path):
    path = str(tmp_path / "s.jsonl")
    storage = await JsonlSessionStorage.create(path, cwd="/tmp")

    await asyncio.gather(*[
        storage.append_message({"role": "user", "content": f"m{i}"})
        for i in range(20)
    ])

    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    # 每行都必须是合法 JSON（并发交错写会产生断行垃圾）
    parsed = [json.loads(ln) for ln in lines]
    assert len(parsed) == 21  # header + 20 messages


async def test_mixed_append_types_form_linear_chain(tmp_path):
    path = str(tmp_path / "s.jsonl")
    storage = await JsonlSessionStorage.create(path, cwd="/tmp")

    await asyncio.gather(
        storage.append_message({"role": "user", "content": "m"}),
        storage.append_context([{"op": "test"}], note="n"),
        storage.append_compaction("summary", "", 100),
    )

    entries = storage.entries
    assert len(entries) == 3
    parents = [e.parent_id for e in entries]
    assert parents == [None, entries[0].id, entries[1].id]
