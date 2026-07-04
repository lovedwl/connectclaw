"""RAG subsystem for ConnectClaw — retrieval-augmented generation.

Placed in agent/harness/rag/ so it's available at the harness level,
not just for the coding agent.
"""

from .document_store import Document, DocumentStore
from .embedding_store import EmbeddingStore
from .retriever import Retriever
from .subsystem import RAGSubsystem, RAGConfig

__all__ = [
    "Document",
    "DocumentStore",
    "EmbeddingStore",
    "Retriever",
    "RAGSubsystem",
    "RAGConfig",
]
