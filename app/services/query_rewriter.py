"""
knowledge/query_rewriter.py

Rewrites a natural-language question into short keyword queries
suitable for web search engines — mirrors the ChatGPT rewriting step
in the paper but uses a free Groq model.
"""

from __future__ import annotations

from groq import Groq

from config import REWRITER_MODEL
from services.groq_client import chat


_SYSTEM_PROMPT = """You are a specialized hardware component search query generator.
Given a user query, extract the exact hardware manufacturer part numbers (MPNs) or primary component identifiers.
Output ONLY the component part numbers or keywords separated by spaces — no explanation or punctuation."""

_FEW_SHOT = [
    {
        "role": "user",
        "content": "Are the DRV8301DCA and STM32F405ZGT6 active for new designs, and what is their current warehouse availability?",
    },
    {
        "role": "assistant",
        "content": "DRV8301DCA STM32F405ZGT6",
    },
    {
        "role": "user",
        "content": "Check stock level and pricing for Texas Instruments part DRV8301",
    },
    {
        "role": "assistant",
        "content": "DRV8301",
    },
]



class QueryRewriter:
    """
    Rewrites questions into web-search-friendly keyword queries.

    Usage
    -----
    rewriter = QueryRewriter(groq_client)
    query = rewriter.rewrite("What is the capital of France?")
    # → "capital France; Paris"
    """

    def __init__(self, client: Groq, model: str = REWRITER_MODEL) -> None:
        self.client = client
        self.model = model

    def rewrite(self, question: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_FEW_SHOT,
            {"role": "user", "content": question},
        ]
        raw = chat(self.client, self.model, messages, max_tokens=60, temperature=0.0)
        cleaned = raw.replace(";", " ").strip()
        if not cleaned:
            # Fallback regex extraction of part numbers
            import re
            parts = [t.upper() for t in re.findall(r"\b[A-Za-z0-9\-]{4,}\b", question) if any(c.isdigit() for c in t) and any(c.isalpha() for c in t)]
            return " ".join(parts[:3]) if parts else question
        return cleaned