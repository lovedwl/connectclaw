"""RAG Retriever — query → embed → search → rerank pipeline."""

from __future__ import annotations

from .document_store import DocumentStore
from .embedding_store import EmbeddingStore


class Retriever:
    """End-to-end retrieval pipeline.

    Flow: query → embed → vector search (top_k) → rerank → return top_n

    If no documents ingested, all methods are no-ops returning empty results.
    """

    def __init__(
        self,
        doc_store: DocumentStore,
        embedding_store: EmbeddingStore,
        embedding_provider,
        reranker_provider,
        top_k: int = 20,
        top_n: int = 5,
    ):
        self._doc_store = doc_store
        self._emb_store = embedding_store
        self._embedding = embedding_provider
        self._reranker = reranker_provider
        self._top_k = top_k
        self._top_n = top_n

    @property
    def has_documents(self) -> bool:
        return self._doc_store.has_documents

    async def retrieve(self, query: str) -> list[str]:
        """Retrieve relevant chunks for a query. Returns [] if no documents."""
        if not self.has_documents or not self._emb_store.has_data:
            return []

        # Step 1: Embed query
        query_emb = await self._embedding.embed_query(query)

        # Step 2: Vector search
        candidates = await self._emb_store.search(query_emb, top_k=self._top_k)
        if not candidates:
            return []

        # Step 3: Rerank
        passages = [c["chunk"] for c in candidates]
        ranked = await self._reranker.rerank(query, passages, top_n=self._top_n)

        # Return in order
        results = []
        for idx, score in ranked:
            if idx < len(candidates):
                results.append(candidates[idx]["chunk"])

        return results if results else passages[: self._top_n]

    async def retrieve_formatted(self, query: str) -> str:
        """Retrieve chunks formatted as a context block. Returns '' if empty."""
        chunks = await self.retrieve(query)
        if not chunks:
            return ""
        return "## Relevant Documentation\n\n" + "\n---\n".join(chunks)
