"""
knowledge/query_rewriter.py

Rewrites a natural-language question into short keyword queries
suitable for web search engines — mirrors the ChatGPT rewriting step
in the paper but uses a free Groq model.
"""

from __future__ import annotations

from groq import Groq

from config import REWRITER_MODEL
from groq_client import chat


_SYSTEM_PROMPT = """You are a search-query generator.
Given a question, extract at most three keywords or short phrases separated by semicolons.
These will be used as a web search query. Output ONLY the keywords — no explanation."""

_FEW_SHOT = [
    {
        "role": "user",
        "content": "What is Henry Feilden's occupation?",
    },
    {
        "role": "assistant",
        "content": "Henry Feilden; occupation; politician",
    },
    {
        "role": "user",
        "content": "In what city was Billy Carlson born?",
    },
    {
        "role": "assistant",
        "content": "Billy Carlson; birthplace; city",
    },
    {
        "role": "user",
        "content": "Who was the screenwriter for Death of a Batman?",
    },
    {
        "role": "assistant",
        "content": "Death of a Batman; screenwriter; film",
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
        # Turn semicolons into spaces for the search engine
        return raw.replace(";", " ").strip()