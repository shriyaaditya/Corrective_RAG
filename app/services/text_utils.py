"""
utils/text_utils.py
Helpers for splitting documents into knowledge strips.
"""
import re


def split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter (no NLTK dependency)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_into_strips(text: str, strip_size: int = 3) -> list[str]:
    """
    Split *text* into strips of ~strip_size sentences each.
    Very short texts (≤2 sentences) are returned as a single strip.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 2:
        return [text]

    strips = []
    for i in range(0, len(sentences), strip_size):
        chunk = " ".join(sentences[i : i + strip_size])
        strips.append(chunk)
    return strips


def truncate(text: str, max_chars: int = 3000) -> str:
    """Hard-truncate text to avoid blowing the context window."""
    return text[:max_chars] if len(text) > max_chars else text