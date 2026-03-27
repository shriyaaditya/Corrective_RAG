"""
knowledge/knowledge_refiner.py

Implements the *decompose-then-recompose* algorithm from the paper:

  1. Decompose  – split each retrieved document into knowledge strips
  2. Filter     – score each strip against the query; drop irrelevant ones
  3. Recompose  – concatenate surviving strips into a single passage
"""

from __future__ import annotations

from retrieval_evaluator import RetrievalEvaluator
from text_utils import split_into_strips
from config import STRIP_SIZE_SENTENCES, STRIP_FILTER_THRESHOLD, TOP_K_STRIPS


class KnowledgeRefiner:
    """
    Refines a list of retrieved documents into a compact, relevant passage.

    Usage
    -----
    refiner = KnowledgeRefiner(evaluator)
    knowledge = refiner.refine(query, documents)   # → single string
    """

    def __init__(
        self,
        evaluator: RetrievalEvaluator,
        strip_size: int = STRIP_SIZE_SENTENCES,
        filter_threshold: float = STRIP_FILTER_THRESHOLD,
        top_k: int = TOP_K_STRIPS,
    ) -> None:
        self.evaluator = evaluator
        self.strip_size = strip_size
        self.filter_threshold = filter_threshold
        self.top_k = top_k

    def refine(self, query: str, documents: list[str]) -> str:
        """
        Decompose → filter → recompose all documents for *query*.
        Returns a single string of concatenated relevant strips.
        """
        all_strips: list[str] = []
        for doc in documents:
            strips = split_into_strips(doc, strip_size=self.strip_size)
            all_strips.extend(strips)

        if not all_strips:
            return " ".join(documents)

        relevant_strips = self.evaluator.filter_strips(
            query=query,
            strips=all_strips,
            threshold=self.filter_threshold,
            top_k=self.top_k,
        )

        return " ".join(relevant_strips)