"""Tests for BM25 keyword index — the keyword-signal half of hybrid retrieval."""

from __future__ import annotations

from connectclaw.memory.bm25 import BM25Index, _tokenize


def test_tokenize_splits_punctuation_and_lowercases():
    # punctuation splits tokens; underscore stays within a latin token
    assert _tokenize("Hello, World! foo_bar") == ["hello", "world", "foo_bar"]


def test_tokenize_cjk_singles():
    # Chinese has no word delimiter → split into single chars so term overlap
    # is detectable; latin identifiers stay whole.
    assert _tokenize("用户喜欢 ConnectClaw") == [
        "用", "户", "喜", "欢", "connectclaw",
    ]


def test_empty_corpus_returns_empty_scores():
    idx = BM25Index([])
    assert idx.score("anything") == []


def test_empty_query_returns_zeros():
    idx = BM25Index(["some document", "another doc"])
    assert idx.score("") == [0.0, 0.0]


def test_exact_term_match_outranks_unrelated():
    docs = [
        "用 sed 修改文件",
        "重启守护进程",
        "sed 替换文本内容",
    ]
    idx = BM25Index(docs)
    scores = idx.score("sed 修改")
    # docs 0 and 2 mention sed/修改, doc 1 doesn't
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_repeated_terms_score_higher():
    docs = ["foo foo foo bar", "foo bar"]
    idx = BM25Index(docs)
    scores = idx.score("foo")
    assert scores[0] > scores[1]  # higher tf in doc 0


def test_missing_term_scores_zero():
    idx = BM25Index(["alpha beta", "gamma delta"])
    scores = idx.score("zzz")
    assert scores == [0.0, 0.0]


def test_chinese_tokens_match():
    docs = ["用户喜欢深色主题", "系统在凌晨重启"]
    idx = BM25Index(docs)
    scores = idx.score("深色主题")
    assert scores[0] > scores[1]
