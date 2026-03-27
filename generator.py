"""
generator/generator.py

Wraps a Groq-hosted LLM as the final answer generator.
Accepts the query and refined knowledge context, returns the answer.
"""

from __future__ import annotations

from groq import Groq

from config import GENERATOR_MODEL, MAX_TOKENS_GENERATOR, TEMPERATURE_GENERATOR
from groq_client import chat
from text_utils import truncate


_SYSTEM_PROMPT = """You are a knowledgeable assistant.
Answer the question using ONLY the information provided in the CONTEXT.
If the context does not contain enough information, say so briefly — do not fabricate facts.
Be concise and factually precise."""

_USER_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


class Generator:
    """
    Final answer generator.

    Usage
    -----
    gen = Generator(groq_client)
    answer = gen.generate(query, knowledge_context)
    """

    def __init__(
        self,
        client: Groq,
        model: str = GENERATOR_MODEL,
        max_tokens: int = MAX_TOKENS_GENERATOR,
        temperature: float = TEMPERATURE_GENERATOR,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, context: str) -> str:
        """Generate an answer for *query* given *context*."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    context=truncate(context, max_chars=4000),
                    question=query,
                ),
            },
        ]
        return chat(
            self.client,
            self.model,
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )