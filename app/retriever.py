from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "are",
    "of",
    "to",
    "in",
    "for",
    "请",
    "问",
    "的",
    "了",
    "是",
    "吗",
    "呢",
    "我",
    "你",
}


@dataclass(frozen=True)
class Chunk:
    id: int
    document_id: int
    title: str
    content: str


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


def _tfidf_vectors(query: str, chunks: list[Chunk]) -> tuple[Counter[str], list[Counter[str]], dict[str, float]]:
    query_tf = Counter(tokenize(query))
    doc_tfs = [Counter(tokenize(chunk.content)) for chunk in chunks]
    doc_count = max(1, len(doc_tfs))

    document_frequency: Counter[str] = Counter()
    for tf in doc_tfs:
        document_frequency.update(tf.keys())

    idf = {
        token: math.log((doc_count + 1) / (df + 1)) + 1
        for token, df in document_frequency.items()
    }
    return query_tf, doc_tfs, idf


def _cosine(query_tf: Counter[str], doc_tf: Counter[str], idf: dict[str, float]) -> float:
    tokens = set(query_tf) | set(doc_tf)
    if not tokens:
        return 0.0

    dot = 0.0
    query_norm = 0.0
    doc_norm = 0.0

    for token in tokens:
        weight = idf.get(token, 1.0)
        query_weight = query_tf.get(token, 0) * weight
        doc_weight = doc_tf.get(token, 0) * weight
        dot += query_weight * doc_weight
        query_norm += query_weight * query_weight
        doc_norm += doc_weight * doc_weight

    if query_norm == 0.0 or doc_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(query_norm) * math.sqrt(doc_norm))


def retrieve(query: str, chunks: list[Chunk], top_k: int = 4) -> list[RetrievalHit]:
    if not query.strip() or not chunks:
        return []

    query_tf, doc_tfs, idf = _tfidf_vectors(query, chunks)
    hits = [
        RetrievalHit(chunk=chunk, score=_cosine(query_tf, doc_tf, idf))
        for chunk, doc_tf in zip(chunks, doc_tfs, strict=True)
    ]
    return [hit for hit in sorted(hits, key=lambda h: h.score, reverse=True)[:top_k] if hit.score > 0]

