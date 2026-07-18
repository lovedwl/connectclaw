"""Lightweight KMeans for memory consolidation (the "dreaming" dedup step).

Used to cluster semantically similar memories before merging, so the LLM
(or a deterministic fallback) only has to reconcile a small in-cluster set
instead of the full history. Avoids the "全量 100 条丢给大模型" problem.

Pure numpy — no sklearn dependency. Deterministic via ``seed`` so tests are
stable. Cosine distance (vectors are L2-normalized internally) which matches
how the rest of the memory system scores similarity.
"""

from __future__ import annotations

import numpy as np


def kmeans(
    data,
    k: int,
    *,
    seed: int = 0,
    max_iter: int = 50,
) -> np.ndarray:
    """Lloyd's KMeans on rows of ``data``, cosine distance, fixed seed.

    Returns an int array of cluster labels, len == number of rows.
    """
    data = np.asarray(data, dtype=np.float32)
    n = data.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    k = max(1, min(k, n))

    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = data / norms

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = unit[init_idx].copy()

    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        cnorms = np.linalg.norm(centroids, axis=1, keepdims=True)
        cnorms[cnorms == 0] = 1.0
        cunit = centroids / cnorms
        sims = unit @ cunit.T  # (n, k) cosine similarity
        new_labels = sims.argmax(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = unit[mask].mean(axis=0)
    return labels
