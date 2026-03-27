"""
retriever/vector_retriever.py

Drop-in replacement for CorpusRetriever that queries FAISS instead of TF-IDF.

The interface is identical — .retrieve(query, top_k) returns a list of
Document objects — so crag_pipeline.py needs only a one-line change.

Usage
-----
    store     = VectorStore(store_dir="vector_db")
    retriever = VectorRetriever(store)
    docs      = retriever.retrieve("what is retrieval augmented generation?", top_k=5)
    for doc in docs:
        print(doc.text[:120])
"""

from __future__ import annotations

from courpus_retrieval import Document   # reuse the same dataclass
from retriever.vector_store import VectorStore


class VectorRetriever:
    """
    Semantic retriever backed by a VectorStore (FAISS + sentence-transformers).

    Parameters
    ----------
    store : VectorStore
        The loaded (and possibly already populated) vector store.
    score_threshold : float
        Minimum cosine similarity score for a result to be returned.
        Keeps irrelevant chunks out even if top_k results are requested.
        Range [0, 1]; default 0.15 is intentionally permissive.
    """

    def __init__(
        self,
        store: VectorStore,
        score_threshold: float = 0.15,
    ) -> None:
        self.store           = store
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Return the top-k most relevant Document objects for *query*.

        Returns an empty list if the store is empty or no results
        exceed the score threshold.
        """
        if self.store.total_chunks() == 0:
            print("[VectorRetriever] Vector store is empty — no results.")
            return []

        raw_results = self.store.search(query, top_k=top_k)

        docs = []
        for i, (text, score) in enumerate(raw_results):
            if score < self.score_threshold:
                continue
            docs.append(
                Document(
                    id=str(i),
                    text=text,
                    metadata={"score": round(score, 4)},
                )
            )

        return docs

    def __len__(self) -> int:
        return self.store.total_chunks()