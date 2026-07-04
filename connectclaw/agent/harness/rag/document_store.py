"""Document ingestion and chunking for RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A document ingested into the RAG store."""

    path: str
    content: str = ""
    chunks: list[str] = field(default_factory=list)
    chunk_count: int = 0


class DocumentStore:
    """Ingest and chunk documents from the filesystem.

    Chunking strategy: ~500 token chunks with 50 token overlap.
    Uses markdown-aware splitter for .md files, simple text splitter
    for everything else.
    """

    CHUNK_SIZE = 500   # target tokens per chunk
    CHUNK_OVERLAP = 50  # token overlap between chunks

    def __init__(self):
        self._documents: dict[str, Document] = {}

    @property
    def has_documents(self) -> bool:
        return len(self._documents) > 0

    @property
    def documents(self) -> dict[str, Document]:
        return dict(self._documents)

    async def ingest_file(self, path: str) -> Document | None:
        """Read and chunk a single file. Returns None if not readable."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            return None

        chunks = self._chunk(content, os.path.splitext(path)[1])
        doc = Document(
            path=path,
            content=content,
            chunks=chunks,
            chunk_count=len(chunks),
        )
        self._documents[path] = doc
        return doc

    async def ingest_directory(self, path: str, recursive: bool = True) -> list[Document]:
        """Recursively ingest a directory tree. Skips binary files."""
        docs: list[Document] = []
        target = Path(path)

        if not target.exists():
            return docs

        pattern = "**/*" if recursive else "*"
        for file_path in target.glob(pattern):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()
            if ext in _SKIP_EXTENSIONS:
                continue
            doc = await self.ingest_file(str(file_path))
            if doc:
                docs.append(doc)

        return docs

    def remove_document(self, path: str) -> None:
        """Remove a document and its chunks."""
        self._documents.pop(path, None)

    def clear(self) -> None:
        self._documents.clear()

    def _chunk(self, content: str, extension: str) -> list[str]:
        """Split content into overlapping chunks."""
        if not content.strip():
            return []

        # Rough token estimate: chars / 4
        chars_per_chunk = self.CHUNK_SIZE * 4
        chars_overlap = self.CHUNK_OVERLAP * 4

        # For markdown, try to split on ## headings
        if extension in (".md", ".markdown"):
            return self._chunk_markdown(content, chars_per_chunk, chars_overlap)

        # Default: split on paragraphs then combine
        return self._chunk_text(content, chars_per_chunk, chars_overlap)

    def _chunk_markdown(self, content: str, chunk_size: int, overlap: int) -> list[str]:
        """Chunk markdown, splitting on ## headings when possible."""
        sections = content.split("\n## ")
        chunks = []

        for section in sections:
            if not section.strip():
                continue
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Sub-split long sections
                sub = self._chunk_text(section, chunk_size, overlap)
                chunks.extend(sub)

        return chunks

    def _chunk_text(self, content: str, chunk_size: int, overlap: int) -> list[str]:
        """Chunk plain text by paragraphs with overlap."""
        paragraphs = content.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            if current_len + para_len > chunk_size and current:
                chunks.append("\n\n".join(current))
                # Keep overlap: retain last paragraph
                if len(current) > 1:
                    current = [current[-1]]
                    current_len = len(current[-1])
                else:
                    current = []
                    current_len = 0

            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        return chunks


# File extensions to skip during ingestion
_SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".exe", ".bin", ".dat", ".db", ".sqlite",
    ".o", ".a", ".class", ".jar",
    ".woff", ".woff2", ".ttf", ".eot",
    ".wasm", ".onnx", ".pt", ".pth",
}
