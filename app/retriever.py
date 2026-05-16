from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.vector_retriever import vector_scores


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
NOISE_EVIDENCE_TERMS = {
    "一下",
    "可以",
    "这个",
    "那个",
    "东西",
    "怎么",
    "怎么办",
    "需要",
    "能不能",
}


@dataclass(frozen=True)
class Chunk:
    id: int
    document_id: int
    title: str
    content: str
    position: int = 0


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    context: str = ""
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExternalRetrievalScore:
    score: float
    evidence: tuple[str, ...] = ()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


def extract_clues(query: str) -> list[str]:
    """Extract exact lexical clues before falling back to token scoring."""
    phrases = re.findall(r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fff]{2,}", query.lower())
    tokens = [
        token
        for token in tokenize(query)
        if len(token) > 1 or re.fullmatch(r"[a-zA-Z0-9_]+", token)
    ]

    clues: list[str] = []
    for item in [*phrases, *tokens]:
        if item not in clues:
            clues.append(item)
    return clues


def _common_chinese_spans(clue: str, target: str, limit: int = 3) -> list[str]:
    if not re.fullmatch(r"[\u4e00-\u9fff]+", clue):
        return []

    spans: list[str] = []
    occupied: set[int] = set()
    for size in range(len(clue), 1, -1):
        for start in range(0, len(clue) - size + 1):
            span = clue[start : start + size]
            if span not in target:
                continue
            if any(index in occupied for index in range(start, start + size)):
                continue

            spans.append(span)
            occupied.update(range(start, start + size))
            if len(spans) >= limit:
                return spans
    return spans


def _matched_clues(clue: str, text: str) -> list[str]:
    clue_lower = clue.lower()
    text_lower = text.lower()
    if clue_lower in text_lower:
        return [clue]
    return [
        span
        for span in _common_chinese_spans(clue, text_lower)
        if span not in NOISE_EVIDENCE_TERMS
    ]


def search_corpus(
    clues: list[str],
    chunks: list[Chunk],
) -> dict[int, tuple[float, tuple[str, ...]]]:
    """Search raw chunks for exact clue matches, similar to grep over a corpus."""
    matches: dict[int, tuple[float, tuple[str, ...]]] = {}
    for clue in clues:
        for chunk in chunks:
            content_matches = _matched_clues(clue, chunk.content)
            title_matches = _matched_clues(clue, chunk.title)
            matched_terms = [*content_matches, *title_matches]
            if not matched_terms:
                continue

            weight = 2.0 if content_matches else 1.0
            if max(len(term) for term in matched_terms) >= 4:
                weight += 0.8
            current_score, current_evidence = matches.get(chunk.id, (0.0, ()))
            evidence = current_evidence
            for term in matched_terms:
                reason = f"原文命中 `{term}`"
                if reason not in evidence:
                    evidence = (*evidence, reason)
            matches[chunk.id] = (current_score + weight, evidence)
    return matches


def read_context(chunk: Chunk, chunks: list[Chunk], window: int = 1) -> str:
    """Read neighboring chunks from the same document for local context checks."""
    neighbors = [
        item
        for item in chunks
        if item.document_id == chunk.document_id and abs(item.position - chunk.position) <= window
    ]
    neighbors.sort(key=lambda item: item.position)
    return "\n\n".join(item.content for item in neighbors)


def _tfidf_vectors(
    query: str,
    chunks: list[Chunk],
) -> tuple[Counter[str], list[Counter[str]], dict[str, float]]:
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


def retrieve(
    query: str,
    chunks: list[Chunk],
    top_k: int = 4,
    external_scores: dict[int, ExternalRetrievalScore] | None = None,
) -> list[RetrievalHit]:
    if not query.strip() or not chunks:
        return []

    external_scores = external_scores or {}
    clues = extract_clues(query)
    exact_matches = search_corpus(clues, chunks)
    semantic_scores = vector_scores(
        query,
        [(chunk.id, f"{chunk.title}\n{chunk.content}") for chunk in chunks],
    )
    query_tf, doc_tfs, idf = _tfidf_vectors(query, chunks)
    hits: list[RetrievalHit] = []
    for chunk, doc_tf in zip(chunks, doc_tfs, strict=True):
        lexical_score, evidence = exact_matches.get(chunk.id, (0.0, ()))
        external_score = external_scores.get(chunk.id, ExternalRetrievalScore(0.0))
        tfidf_score = _cosine(query_tf, doc_tf, idf)
        semantic_score = semantic_scores.get(chunk.id, 0.0)
        if (
            lexical_score == 0.0
            and external_score.score == 0.0
            and tfidf_score == 0.0
            and semantic_score < 0.08
        ):
            continue

        combined_score = lexical_score + external_score.score + tfidf_score + semantic_score * 1.2
        for reason in external_score.evidence:
            if reason not in evidence:
                evidence = (*evidence, reason)
        if tfidf_score > 0.0:
            evidence = (*evidence, "TF-IDF 弱线索召回")
        if semantic_score >= 0.08:
            evidence = (*evidence, "向量语义召回")
        evidence = evidence[:8]
        hits.append(
            RetrievalHit(
                chunk=chunk,
                score=combined_score,
                context=read_context(chunk, chunks),
                evidence=evidence,
            )
        )

    sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
    return [hit for hit in sorted_hits[:top_k] if hit.score > 0]
