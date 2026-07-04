"""RAG Subsystem — assembly point for the RAG pipeline.

All components are lazily initialized. If not configured (no documents),
all methods are no-ops.
"""

from __future__ import annotations

from dataclasses import dataclass

from connectclaw.logging import get_logger

from .document_store import DocumentStore
from .embedding_store import EmbeddingStore
from .retriever import Retriever

logger = get_logger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG subsystem."""

    enabled: bool = False
    docs_dir: str = ""
    db_path: str = "~/.connectclaw/rag_db"
    top_k: int = 20
    top_n: int = 5


class RAGSubsystem:
    """Optional RAG subsystem.

    Usage:
        rag = RAGSubsystem(config)
        await rag.initialize()  # ingest docs if configured
        context = await rag.search("how does auth work?")
        # If no documents, context is "" and nothing is loaded.
    """

    def __init__(self, config: RAGConfig):
        self._config = config
        self._doc_store = DocumentStore()
        self._embedding_provider = None
        self._reranker_provider = None
        self._emb_store: EmbeddingStore | None = None
        self._retriever: Retriever | None = None
        self._initialized = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled and bool(self._config.docs_dir)

    @property
    def has_documents(self) -> bool:
        return self._doc_store.has_documents

    async def initialize(self) -> None:
        """Initialize the RAG subsystem. No-op if not configured."""
        if self._initialized:
            return

        if not self.enabled:
            self._initialized = True
            return

        import os

        # Lazy import heavy ML dependencies
        try:
            from connectclaw.provider.embedding import EmbeddingProvider
            from connectclaw.provider.rerank import RerankerProvider

            self._embedding_provider = EmbeddingProvider()
            self._reranker_provider = RerankerProvider()

            db_path = os.path.expanduser(self._config.db_path)
            self._emb_store = EmbeddingStore(db_path, self._embedding_provider)

            self._retriever = Retriever(
                self._doc_store,
                self._emb_store,
                self._embedding_provider,
                self._reranker_provider,
                top_k=self._config.top_k,
                top_n=self._config.top_n,
            )

            # Ingest documents
            docs_dir = os.path.expanduser(self._config.docs_dir)
            docs = await self._doc_store.ingest_directory(docs_dir)
            for doc in docs:
                await self._emb_store.add_document(doc)

            logger.info("RAG initialized: %d documents indexed", len(docs))
        except ImportError as e:
            logger.warning("RAG not available (missing dependencies): %s", e)
        except Exception as e:
            logger.error("RAG initialization failed: %s", e)

        self._initialized = True

    async def search(self, query: str) -> str:
        """Search RAG for relevant context. Returns formatted string or ''."""
        if not self._initialized:
            await self.initialize()

        if self._retriever is None or not self.has_documents:
            return ""

        return await self._retriever.retrieve_formatted(query)

    async def add_document(self, path: str) -> None:
        """Add or update a single document."""
        if not self._initialized:
            await self.initialize()

        doc = await self._doc_store.ingest_file(path)
        if doc and self._emb_store:
            await self._emb_store.add_document(doc)

    async def add_directory(self, path: str) -> list[str]:
        """Add all documents from a directory. Returns list of added file paths."""
        if not self._initialized:
            await self.initialize()

        docs = await self._doc_store.ingest_directory(path)
        if self._emb_store:
            for doc in docs:
                await self._emb_store.add_document(doc)

        return [d.path for d in docs]

    async def remove_document(self, path: str) -> None:
        """Remove a document from the store."""
        self._doc_store.remove_document(path)
        if self._emb_store:
            await self._emb_store.remove_document(path)

    async def clear(self) -> None:
        """Clear all documents."""
        self._doc_store.clear()
        if self._emb_store:
            await self._emb_store.clear()
