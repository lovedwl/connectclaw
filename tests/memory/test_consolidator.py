"""Tests for clustering-based memory consolidation (the "dreaming" dedup).

How semantically similar memories are grouped before merging, so the LLM
(or deterministic fallback) reconciles a small cluster instead of being
fed 100 memories at once.
"""

from __future__ import annotations

import pytest

from connectclaw.memory.clustering import kmeans
from connectclaw.memory.consolidator import ConsolidationConfig, MemoryConsolidator
from connectclaw.memory.store import MemoryStore
from connectclaw.memory.types import MemoryEntry, MemoryType


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    yield s
    s.close()


@pytest.fixture
def consolidator(store):
    return MemoryConsolidator(store, ConsolidationConfig())


def _vec(seed: int) -> list[float]:
    """Two-group separable vector: group A around (1,0,0), group B around (0,1,0)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    base = [0.0, 0.0, 0.0]
    base[seed % 2] = 1.0
    return (base + rng.normal(0, 0.05, 3)).tolist()


def _make(store, content, embedding=None, strength=1.0,
          mtype=MemoryType.PROCEDURAL):
    e = MemoryEntry(
        type=mtype,
        content=content,
        strength=strength,
        embedding=embedding or [],
    )
    store.add(e)
    return e


# ── kmeans unit ──────────────────────────────────────────────

def test_kmeans_separates_two_groups():
    import numpy as np
    group_a = np.array([[1, 0, 0], [0.95, 0.05, 0], [0.9, 0.1, 0]], dtype=np.float32)
    group_b = np.array([[0, 1, 0], [0.05, 0.95, 0], [0.1, 0.9, 0]], dtype=np.float32)
    data = np.vstack([group_a, group_b])
    labels = kmeans(data, 2, seed=0)
    assert len(labels) == 6
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_kmeans_deterministic():
    import numpy as np
    rng = np.random.default_rng(1)
    data = rng.normal(size=(20, 8)).astype(np.float32)
    a = kmeans(data, 4, seed=7)
    b = kmeans(data, 4, seed=7)
    assert list(a) == list(b)


def test_kmeans_empty():
    import numpy as np
    assert list(kmeans(np.zeros((0, 3)), 3)) == []


def test_kmeans_k_larger_than_n():
    import numpy as np
    data = np.array([[1, 0], [0, 1]], dtype=np.float32)
    labels = kmeans(data, 10)  # k clamped to n
    assert set(labels.tolist()) <= {0, 1}


# ── consolidator clustering + merge ──────────────────────────

def test_cluster_memories_groups_similar(store, consolidator):
    a1 = _make(store, "用 sed 改文件", _vec(0))
    a2 = _make(store, "用 sed 替换", _vec(2))
    b1 = _make(store, "重启服务", _vec(1))
    b2 = _make(store, "重启守护进程", _vec(3))

    clusters = consolidator.cluster_memories([a1, a2, b1, b2], k=2)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 2]
    # a1 and a2 should land together
    cluster_ids = [{e.id for e in c} for c in clusters]
    assert any({a1.id, a2.id} == c for c in cluster_ids)


def test_cluster_memories_handles_missing_embeddings(store, consolidator):
    with_vec = _make(store, "has vec", _vec(0))
    no_vec = _make(store, "no vec")
    clusters = consolidator.cluster_memories([with_vec, no_vec], k=2)
    # no_vec becomes its own singleton
    assert any(c == [no_vec] for c in clusters)


def test_consolidate_by_clustering_merges_and_deletes(store, consolidator):
    a1 = _make(store, "sed 改文件", _vec(0), strength=0.8)
    a2 = _make(store, "sed 替换", _vec(2), strength=0.7)
    a3 = _make(store, "sed 修改", _vec(4), strength=0.6)
    b1 = _make(store, "重启服务", _vec(1), strength=0.9)

    merged = consolidator.consolidate_by_clustering([a1, a2, a3, b1], k=2)
    assert merged == 2  # a-cluster lost 2, b-cluster singleton lost 0
    remaining = store.list_all()
    assert len(remaining) == 2
    # kept memory should contain folded content
    kept_a = next(e for e in remaining if "sed" in e.content)
    assert "sed 替换" in kept_a.content or "sed 修改" in kept_a.content


def test_consolidate_singletons_left_alone(store, consolidator):
    e = _make(store, "lone", _vec(0))
    n = consolidator.consolidate_by_clustering([e], k=2)
    assert n == 0
    assert store.get(e.id) is not None
    assert store.get(e.id).content == "lone"


def test_consolidate_strengthens_keeper(store, consolidator):
    a1 = _make(store, "a1", _vec(0), strength=0.5)
    a2 = _make(store, "a2", _vec(2), strength=0.5)
    a3 = _make(store, "a3", _vec(4), strength=0.5)
    before = store.get(a1.id).strength
    consolidator.consolidate_by_clustering([a1, a2, a3], k=1)
    after = store.get(a1.id).strength
    assert after > before


def test_consolidate_does_not_merge_across_clusters(store, consolidator):
    a = _make(store, "sed 改文件", _vec(0))
    b = _make(store, "重启服务", _vec(1))
    # k=2 keeps them in separate clusters → nothing merged
    n = consolidator.consolidate_by_clustering([a, b], k=2)
    assert n == 0
    assert len(store.list_all()) == 2
