"""
evaluator/retrieval_evaluator.py

Replaces the T5-large retrieval evaluator from the paper with a Groq-hosted
LLM.  The evaluator scores each (query, document) pair and returns a float
in [-1, 1].

Scoring rubric
--------------
  1.0  → document clearly answers the question          (CORRECT)
  0.0  → document is partially / tangentially relevant  (AMBIGUOUS)
 -1.0  → document is unrelated / misleading             (INCORRECT)
"""

from __future__ import annotations

from groq import Groq

from config import EVALUATOR_MODEL, UPPER_THRESHOLD, LOWER_THRESHOLD
from groq_client import chat
from text_utils import truncate


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a strict relevance evaluator for a retrieval-augmented system.

Given a QUESTION and a DOCUMENT, decide how well the document can answer the question.

Reply with ONLY one of these three tokens — nothing else:
  CORRECT    – the document contains information that directly answers the question
  INCORRECT  – the document is unrelated or misleading
  AMBIGUOUS  – the document is partially related but does not fully answer the question
"""

_USER_TEMPLATE = """QUESTION: {question}

DOCUMENT: {document}

Relevance verdict (CORRECT / INCORRECT / AMBIGUOUS):"""


# ── Token → score mapping ─────────────────────────────────────────────────────

_LABEL_TO_SCORE: dict[str, float] = {
    "CORRECT": 1.0,
    "AMBIGUOUS": 0.0,
    "INCORRECT": -1.0,
}


def _parse_score(raw: str) -> float:
    """Extract a numeric score from the model output."""
    upper = raw.upper().strip()
    for label, score in _LABEL_TO_SCORE.items():
        if label in upper:
            return score
    # Fallback: treat unexpected output as ambiguous
    return 0.0


# ── Public API ────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Lightweight retrieval evaluator backed by a Groq-hosted LLM.

    Usage
    -----
    evaluator = RetrievalEvaluator(groq_client)
    score = evaluator.score(query, document)          # single pair → float
    scores = evaluator.score_batch(query, documents)  # list of docs → list[float]
    confidence = evaluator.judge(query, documents)    # → "CORRECT" | "INCORRECT" | "AMBIGUOUS"
    """

    def __init__(self, client: Groq, model: str = EVALUATOR_MODEL) -> None:
        self.client = client
        self.model = model

    def score(self, query: str, document: str) -> float:
        """Return relevance score in [-1, 1] for a single (query, doc) pair."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    question=query,
                    document=truncate(document, max_chars=2000),
                ),
            },
        ]
        raw = chat(self.client, self.model, messages, max_tokens=10, temperature=0.0)
        return _parse_score(raw)

    def score_batch(self, query: str, documents: list[str]) -> list[float]:
        """Score multiple documents for the same query."""
        return [self.score(query, doc) for doc in documents]

    def judge(
        self,
        query: str,
        documents: list[str],
        upper: float = UPPER_THRESHOLD,
        lower: float = LOWER_THRESHOLD,
    ) -> str:
        """
        Aggregate document scores into a single action label.

        Returns
        -------
        "CORRECT"   if any document score ≥ upper
        "INCORRECT" if all document scores < lower
        "AMBIGUOUS" otherwise
        """
        scores = self.score_batch(query, documents)
        if any(s >= upper for s in scores):
            return "CORRECT"
        if all(s < lower for s in scores):
            return "INCORRECT"
        return "AMBIGUOUS"

    def filter_strips(
        self,
        query: str,
        strips: list[str],
        threshold: float,
        top_k: int,
    ) -> list[str]:
        """
        Score every knowledge strip and keep the top-k above *threshold*.
        Returns strips in their original order.
        """
        scored = [(strip, self.score(query, strip)) for strip in strips]
        passed = [(s, sc) for s, sc in scored if sc >= threshold]
        # Sort by score descending, then take top-k
        passed.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in passed[:top_k]]
        # Restore original order
        original_order = [s for s in strips if s in selected]
        return original_order if original_order else [strips[0]]  # always return ≥1