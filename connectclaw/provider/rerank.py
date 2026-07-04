"""BGE-Reranker-v2-m3 — lazy-loaded local reranker model."""

from __future__ import annotations

import asyncio


class RerankerProvider:
    """Lazy-loaded BGE-Reranker-v2-m3.

    If RAG is not configured, this class is never instantiated.
    On first use, downloads the model from HuggingFace.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ):
        self._model = None
        self._model_name = model_name
        self._device = device

    async def ensure_loaded(self) -> None:
        """Lazy load the model. Called before first rerank()."""
        if self._model is not None:
            return

        from FlagEmbedding import FlagReranker

        self._model = await asyncio.to_thread(
            FlagReranker,
            self._model_name,
            use_fp16=(self._device != "cpu"),
        )

    async def rerank(
        self,
        query: str,
        passages: list[str],
        top_n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Rerank passages. Returns list of (index, score) sorted by score descending.
        """
        await self.ensure_loaded()
        if not passages:
            return []

        pairs = [[query, p] for p in passages]
        scores = await asyncio.to_thread(
            self._model.compute_score,
            pairs,
            normalize=True,
        )

        # Handle single score case
        if not isinstance(scores, list):
            scores = [scores]

        ranked = sorted(
            enumerate(scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return ranked[:top_n]
