"""
knowledge/web_search.py

Free web search using DuckDuckGo (no API key required).
Falls back to Wikipedia-API for authoritative pages when possible.
"""

from __future__ import annotations

import re
import time
import urllib.request
from html.parser import HTMLParser

try:
    from duckduckgo_search import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    _DDG_AVAILABLE = False

try:
    import wikipediaapi
    _WIKI_AVAILABLE = True
except ImportError:
    _WIKI_AVAILABLE = False


# ── Simple HTML stripper ──────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str) -> str:
    parser = _HTMLStripper()
    try:
        parser.feed(html)
        return re.sub(r"\s+", " ", parser.get_text()).strip()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html).strip()


def _fetch_url(url: str, timeout: int = 8) -> str:
    """Download a URL and return stripped plain text."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CRAG-bot/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        return _strip_html(raw)
    except Exception:
        return ""


# ── Wikipedia helper ──────────────────────────────────────────────────────────

def _fetch_wikipedia(title: str) -> str:
    """Return the Wikipedia page summary for *title* (best-effort)."""
    if not _WIKI_AVAILABLE:
        return ""
    try:
        wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="CRAG-bot/1.0"
        )
        page = wiki.page(title)
        if page.exists():
            return page.summary[:3000]
    except Exception:
        pass
    return ""


# ── Public API ────────────────────────────────────────────────────────────────

class WebSearcher:
    """
    Free web searcher.

    Strategy
    --------
    1. Use DuckDuckGo to get result URLs + snippets.
    2. Prefer Wikipedia URLs — fetch full page text via wikipedia-api.
    3. For non-Wikipedia URLs, fetch raw HTML and strip tags.
    4. Return a list of plain-text passages.
    """

    def __init__(self, top_k: int = 5, prefer_wikipedia: bool = True) -> None:
        self.top_k = top_k
        self.prefer_wikipedia = prefer_wikipedia

    def search(self, query: str) -> list[str]:
        """
        Run a web search for *query* and return up to *top_k* text passages.
        """
        results = self._ddg_search(query)
        if not results:
            return []

        passages: list[str] = []
        wiki_results = [r for r in results if "wikipedia.org" in r.get("href", "")]
        other_results = [r for r in results if "wikipedia.org" not in r.get("href", "")]

        # Prioritise Wikipedia
        for r in (wiki_results + other_results)[: self.top_k]:
            href = r.get("href", "")
            body = r.get("body", "")

            if "wikipedia.org/wiki/" in href:
                # Extract article title from URL
                title = href.split("/wiki/")[-1].replace("_", " ")
                text = _fetch_wikipedia(title) or body
            else:
                text = body  # Use DDG snippet; avoids most paywalls

            if text:
                passages.append(text[:3000])

            if len(passages) >= self.top_k:
                break

        return passages

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ddg_search(self, query: str) -> list[dict]:
        if not _DDG_AVAILABLE:
            print("[WebSearcher] duckduckgo-search not installed; skipping web search.")
            return []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.top_k + 3))
            return results
        except Exception as e:
            print(f"[WebSearcher] DuckDuckGo error: {e}")
            return []