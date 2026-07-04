"""BGE-M3 embedding provider — lazy-loaded, local model."""

from __future__ import annotations

import asyncio


class EmbeddingProvider:
    """Lazy-loaded BGE-M3 embedding model.

    If RAG is not configured, this class is never instantiated.
    On first use, downloads the model from HuggingFace.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
    ):
        self._model = None
        self._model_name = model_name
        self._device = device

    async def ensure_loaded(self) -> None:
        """Lazy load the model. Called before first embed()."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._model = await asyncio.to_thread(
            SentenceTransformer,
            self._model_name,
            device=self._device,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        await self.ensure_loaded()
        if not texts:
            return []
        result = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return result.tolist()

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        results = await self.embed([text])
        return results[0] if results else []
