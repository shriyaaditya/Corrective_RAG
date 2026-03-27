"""
retriever/corpus_retriever.py

A simple TF-IDF retriever over a local in-memory corpus.
No heavy ML dependencies — uses only the Python standard library + math.

For production use you could swap this out for a FAISS / BM25 / Elasticsearch
retriever; the rest of the CRAG pipeline stays the same.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ── TF-IDF helpers ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for t in tokens:
        counts[t] += 1
    n = len(tokens) or 1
    return {t: c / n for t, c in counts.items()}


def _idf(term: str, doc_tokens_list: list[list[str]]) -> float:
    df = sum(1 for tokens in doc_tokens_list if term in tokens)
    return math.log((len(doc_tokens_list) + 1) / (df + 1)) + 1


def _tfidf_score(query_tokens: list[str], doc_tokens: list[str], idf_cache: dict[str, float]) -> float:
    tf = _tf(doc_tokens)
    score = sum(idf_cache.get(t, 0) * tf.get(t, 0) for t in query_tokens)
    return score


# ── Retriever ─────────────────────────────────────────────────────────────────

class CorpusRetriever:
    """
    In-memory TF-IDF retriever.

    Usage
    -----
    retriever = CorpusRetriever()
    retriever.add_documents([Document("1", "..."), Document("2", "...")])
    docs = retriever.retrieve("who invented the telephone?", top_k=5)
    """

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._doc_tokens: list[list[str]] = []

    def add_documents(self, documents: list[Document]) -> None:
        for doc in documents:
            self._docs.append(doc)
            self._doc_tokens.append(_tokenize(doc.text))

    def add_texts(self, texts: list[str]) -> None:
        """Convenience: add plain strings as documents."""
        self.add_documents(
            [Document(id=str(i), text=t) for i, t in enumerate(texts)]
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        if not self._docs:
            return []

        query_tokens = _tokenize(query)
        unique_query_terms = set(query_tokens)

        # Build IDF cache for query terms only (much faster than full IDF)
        idf_cache = {
            t: _idf(t, self._doc_tokens) for t in unique_query_terms
        }

        scores = [
            _tfidf_score(query_tokens, dt, idf_cache)
            for dt in self._doc_tokens
        ]

        ranked = sorted(
            range(len(self._docs)), key=lambda i: scores[i], reverse=True
        )
        return [self._docs[i] for i in ranked[:top_k]]

    def __len__(self) -> int:
        return len(self._docs)