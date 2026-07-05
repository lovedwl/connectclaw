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

    # Don't read files larger than this (5 MB) as RAG documents.
    _MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

    async def ingest_file(self, path: str) -> Document | None:
        """Read and chunk a single file. Returns None if not readable,
        too large, or binary."""
        # Check file size first (avoid reading giant files)
        try:
            file_size = os.path.getsize(path)
            if file_size > self._MAX_FILE_SIZE_BYTES:
                return None
        except OSError:
            return None

        # Read first chunk to detect binary content
        try:
            with open(path, "rb") as f:
                head = f.read(8192)
        except OSError:
            return None

        if b"\x00" in head:
            return None  # binary file

        # Decode and read full content
        try:
            content = head.decode("utf-8", errors="replace")
            if file_size > 8192:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(8192)
                    content += f.read()
        except (OSError, UnicodeDecodeError):
            return None

        # Skip files that are mostly garbage after decode
        if _looks_binary(content):
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
        """Recursively ingest a directory tree. Skips binary files and
        known non-document directories (``.venv``, ``.git``, ``target``, etc.)."""
        docs: list[Document] = []
        target = Path(path)

        if not target.exists():
            return docs

        pattern = "**/*" if recursive else "*"
        for file_path in target.glob(pattern):
            if not file_path.is_file():
                continue

            # Skip known non-document directories anywhere in the path
            if _has_excluded_dir(file_path):
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
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".exe", ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".o", ".a", ".class", ".jar", ".pyc",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".wasm", ".onnx", ".pt", ".pth", ".bin",
    ".rlib", ".rmeta", ".pack", ".idx", ".rev", ".keep",
    ".lock",  # package-lock / Cargo.lock are rarely useful docs
    ".ipynb",  # Jupyter notebooks — too much JSON noise for chunking
}

# Directory name components that should never be recursed into
_EXCLUDED_DIR_NAMES = {
    ".venv", "venv", ".env", "env",
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "target",  # Rust build artifacts
    "node_modules", "bower_components",
    ".tox", "eggs", ".eggs", ".egg-info",
    "dist", "build", "__pycache__",
    ".idea", ".vscode", ".vs",
    ".claude",  # Claude Code internal directory
}


def _has_excluded_dir(file_path: Path) -> bool:
    """True if any path component is a known non-document directory."""
    return bool(_EXCLUDED_DIR_NAMES.intersection(file_path.parts))


def _looks_binary(content: str) -> bool:
    """Heuristic: a file with a high ratio of null/replacement chars
    is probably binary that survived ``errors='replace'`` decoding."""
    if len(content) < 32:
        return False
    replacement = content.count("�")  # Unicode replacement char
    nulls = content.count("\x00")
    bad = replacement + nulls
    # If more than 10% of the content is replacement/null chars, call it binary
    return bad > len(content) * 0.10
