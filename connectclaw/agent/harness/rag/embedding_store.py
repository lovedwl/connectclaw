"""LanceDB vector store for document chunks."""

from __future__ import annotations

from .document_store import Document


class EmbeddingStore:
    """LanceDB-based vector store for document chunks.

    Lazy initialization — no connection until first operation.
    Embedding model is provided externally (lazy loaded from provider layer).
    """

    def __init__(self, db_path: str, embedding_provider):
        self._db_path = db_path
        self._embedding = embedding_provider
        self._db = None
        self._table = None
        self._initialized = False

    @property
    def has_data(self) -> bool:
        return self._initialized and self._table is not None

    async def ensure_initialized(self) -> None:
        """Lazy init LanceDB connection and table."""
        if self._initialized:
            return

        import lancedb
        import os

        os.makedirs(self._db_path, exist_ok=True)
        self._db = await lancedb.connect_async(self._db_path)

        try:
            self._table = await self._db.open_table("chunks")
        except Exception:
            self._table = None

        self._initialized = True

    async def add_document(self, doc: Document) -> int:
        """Embed and insert all chunks for a document. Returns number of chunks added."""
        if not doc.chunks:
            return 0

        await self.ensure_initialized()

        # Generate embeddings
        embeddings = await self._embedding.embed(doc.chunks)
        if not embeddings:
            return 0

        rows = [
            {
                "doc_path": doc.path,
                "chunk_index": i,
                "chunk": chunk,
                "vector": emb,
            }
            for i, (chunk, emb) in enumerate(zip(doc.chunks, embeddings))
        ]

        if self._table is None:
            self._table = await self._db.create_table("chunks", rows)
        else:
            await self._table.add(rows)

        return len(rows)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> list[dict]:
        """Vector similarity search. Returns [{doc_path, chunk, _distance}, ...]."""
        if self._table is None:
            return []

        try:
            results = (
                await self._table.search(query_embedding)
                .limit(top_k)
                .to_list()
            )
            return results
        except Exception:
            return []

    async def remove_document(self, path: str) -> None:
        """Remove all chunks for a document."""
        if self._table is not None:
            try:
                await self._table.delete(f"doc_path = '{path}'")
            except Exception:
                pass

    async def clear(self) -> None:
        """Drop all data."""
        if self._table is not None:
            try:
                await self._db.drop_table("chunks")
            except Exception:
                pass
            self._table = None
