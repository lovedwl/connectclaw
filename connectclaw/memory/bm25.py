"""Lightweight BM25 for memory retrieval — keyword-signal complement to embeddings.

Embeddings catch semantic similarity but miss exact-term matches (names, IDs,
file paths, error codes). BM25 covers that. The retriever fuses the two scores
so a memory surfaces whether it matched by meaning or by literal term.

Pure Python, no dependencies. Tokenizer is whitespace + punctuation split —
good enough for memory contents which are already short natural-language lines.
In-memory index rebuilt per query over the candidate set (memories are small,
typically hundreds, so a persistent index isn't worth the complexity).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _tokenize(text: str) -> list[str]:
    """Tokenize for BM25. CJK runs split into single chars (Chinese has no
    word delimiter), latin/underscore runs kept as whole tokens so identifiers
    like ``ConnectClaw`` or ``foo_bar`` stay intact."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        buf: list[str] = []
        for ch in tok:
            if _is_cjk(ch):
                if buf:
                    out.append("".join(buf).lower())
                    buf = []
                out.append(ch)
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf).lower())
    return out


class BM25Index:
    """Okapi BM25 over a fixed corpus, queried by tokenized query."""

    def __init__(self, corpus: list[str], *, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._doc_tokens: list[list[str]] = [_tokenize(doc) for doc in corpus]
        self._doc_len = [len(d) for d in self._doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

        self._df: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._df[term] += 1
        self._n = len(self._doc_tokens)

        # term -> (doc_idx, tf) postings
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for i, tokens in enumerate(self._doc_tokens):
            tf = Counter(tokens)
            for term, freq in tf.items():
                self._postings.setdefault(term, []).append((i, freq))

    def score(self, query: str) -> list[float]:
        """Return one BM25 score per corpus doc, in corpus order."""
        q_terms = _tokenize(query)
        scores = [0.0] * self._n
        if not q_terms or self._n == 0:
            return scores

        for term in q_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = self._df[term]
            idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            for doc_idx, tf in postings:
                dl = self._doc_len[doc_idx] or 1
                denom = tf + self._k1 * (
                    1 - self._b + self._b * (dl / (self._avgdl or 1))
                )
                scores[doc_idx] += idf * (tf * (self._k1 + 1)) / denom
        return scores
