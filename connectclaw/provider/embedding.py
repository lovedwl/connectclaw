"""BGE-M3 embedding provider — lazy-loaded, local model."""

from __future__ import annotations

import asyncio


def _resolve_device() -> str:
    """Pick 'cuda' when a GPU is available, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class EmbeddingProvider:
    """Lazy-loaded BGE-M3 embedding model.

    If RAG is not configured, this class is never instantiated.
    On first use, downloads the model from HuggingFace.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
    ):
        self._model = None
        self._model_name = model_name
        # Auto-pick GPU when available — BGE-M3 on CPU is ~10x slower per query,
        # and query embedding sits on the per-turn recall path.
        self._device = device or _resolve_device()

    async def ensure_loaded(self) -> None:
        """Lazy load the model. Called before first embed()."""
        if self._model is not None:
            return

        # Hard-cap torch CPU threads (belt-and-braces over OMP_NUM_THREADS):
        # some paths ignore the env var. On CPU, BGE-M3 single-query embedding
        # does not benefit from all cores and an unbounded pool spikes RSS.
        try:
            import os as _os
            import torch

            n = int(_os.environ.get("CONNECTCLAW_ML_THREADS", "4"))
            torch.set_num_threads(max(1, n))
        except Exception:
            pass

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


# ── Shared instance ────────────────────────────────────────────

_shared_provider: EmbeddingProvider | None = None


def get_shared_embedding_provider(
    model_name: str = "BAAI/bge-m3", device: str | None = None
) -> EmbeddingProvider:
    """Return a process-wide shared EmbeddingProvider.

    RAG and the memory subsystem both need BGE-M3. Sharing one instance
    avoids loading the ~2GB model into memory twice. Device is auto-detected
    (GPU if available) unless explicitly given.
    """
    global _shared_provider
    if _shared_provider is None:
        _shared_provider = EmbeddingProvider(model_name=model_name, device=device)
    return _shared_provider
