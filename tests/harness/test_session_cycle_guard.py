"""2026-10-16 regression tests — get_path_to_root must never RecursionError.

Incident: after /stop cancelled a mid-flight turn while a subagent fleet was
still appending, the in-memory session entry graph got corrupted; the
recursive DFS in get_path_to_root() blew the Python stack within
milliseconds, and EVERY later message in the chat died with
"maximum recursion depth exceeded" (recall / context_state / memory
extraction / pre-/new extraction all traverse this graph). /new only escaped
it by switching to a fresh session. Fix: iterative DFS + visited set —
self-loops, detached cycles and duplicate-id reference chains all terminate.
"""

from __future__ import annotations

import sys

from connectclaw.agent.harness.session import JsonlSessionStorage


async def _mk(tmp_path, n=3):
    path = str(tmp_path / "s.jsonl")
    storage = await JsonlSessionStorage.create(path, cwd="/tmp")
    for i in range(n):
        await storage.append_message({"role": "user", "content": f"m{i}"})
    return storage


async def _run_capped(storage):
    """Call get_path_to_root with a low recursion limit — a runaway
    traversal dies instantly instead of eating 1000 frames."""
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(150)
    try:
        return await storage.get_path_to_root()
    finally:
        sys.setrecursionlimit(old)


async def test_self_loop_terminates(tmp_path):
    storage = await _mk(tmp_path)
    e2 = storage._entries[2]
    e2.parent_id = e2.id  # self-loop → entry becomes unreachable, no crash
    entries = await _run_capped(storage)
    assert [e.id for e in entries] == [e.id for e in storage.entries[:2]]


async def test_detached_cycle_terminates(tmp_path):
    storage = await _mk(tmp_path)
    e1, e2 = storage._entries[1], storage._entries[2]
    e1.parent_id, e2.parent_id = e2.id, e1.id  # closed loop
    entries = await _run_capped(storage)
    assert len(entries) >= 1  # terminates; loop entries unreachable


async def test_duplicate_id_parent_chain_terminates(tmp_path):
    """The truly fatal shape for the old recursive walk: two entries share
    one id and one of them references that id as its parent — walk() then
    recursed through _by_id[id] → children[id] forever."""
    storage = await _mk(tmp_path)
    e0, e1, e2 = storage._entries
    e2.id = e1.id              # id collision (as after a torn/corrupted append)
    e2.parent_id = e1.id       # points at the shared id → old walk: ∞ recursion
    storage._by_id[e2.id] = e2  # dict semantics: later object wins
    entries = await _run_capped(storage)
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))  # each id visited at most once


async def test_dangling_parent_reported_not_visited(tmp_path):
    storage = await _mk(tmp_path, n=2)
    storage._entries[1].parent_id = "deadbeefdead"  # parent does not exist
    entries = await _run_capped(storage)
    assert [e.id for e in entries] == [storage.entries[0].id]


async def test_normal_chain_order_preserved(tmp_path):
    storage = await _mk(tmp_path, n=5)
    ids = [await storage.append_message({"role": "user", "content": f"m{i}"})
           for i in range(3)]
    entries = await _run_capped(storage)
    assert [e.id for e in entries if e.type == "message"][-3:] == ids
