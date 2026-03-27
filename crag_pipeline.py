"""
crag_pipeline.py

Main CRAG orchestrator — implements Algorithm 1 from the paper.

┌──────────────────────────────────────────────────────────────────────┐
│  Input: query + local corpus documents                               │
│                                                                      │
│  1. Retrieve top-K documents from local corpus                       │
│  2. Evaluate relevance → action ∈ {CORRECT, INCORRECT, AMBIGUOUS}   │
│  3a. CORRECT  → knowledge refinement (decompose-filter-recompose)    │
│  3b. INCORRECT→ web search + external knowledge selection            │
│  3c. AMBIGUOUS→ both 3a and 3b combined                              │
│  4. Generate final answer with refined context                       │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from groq import Groq

from config import TOP_K_DOCUMENTS, TOP_K_WEB_RESULTS
from retrieval_evaluator import RetrievalEvaluator
from generator import Generator
from knowledge_refiner import KnowledgeRefiner
from query_rewriter import QueryRewriter
from web_search import WebSearcher
from courpus_retrieval import CorpusRetriever, Document
from retriever.vector_retriever import VectorRetriever
from groq_client import get_client

# Accept either retriever — VectorRetriever is the new default
AnyRetriever = CorpusRetriever | VectorRetriever


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CRAGResult:
    query: str
    action: str                  # "CORRECT" | "INCORRECT" | "AMBIGUOUS"
    internal_knowledge: Optional[str]
    external_knowledge: Optional[str]
    final_context: str
    answer: str
    retrieved_docs: list[str]


# ── Pipeline ──────────────────────────────────────────────────────────────────

class CRAGPipeline:
    """
    End-to-end CRAG pipeline.

    Parameters
    ----------
    corpus_retriever : CorpusRetriever
        Pre-loaded local document store.
    groq_client : Groq | None
        Shared Groq client (created automatically if None).
    top_k_docs : int
        How many documents to retrieve from the local corpus.
    top_k_web : int
        How many web passages to fetch when doing web search.
    verbose : bool
        Print step-by-step progress.
    """

    def __init__(
        self,
        corpus_retriever: "AnyRetriever",
        groq_client: Optional[Groq] = None,
        top_k_docs: int = TOP_K_DOCUMENTS,
        top_k_web: int = TOP_K_WEB_RESULTS,
        verbose: bool = True,
    ) -> None:
        self.retriever = corpus_retriever
        self.client = groq_client or get_client()
        self.top_k_docs = top_k_docs
        self.top_k_web = top_k_web
        self.verbose = verbose

        # Sub-components
        self.evaluator = RetrievalEvaluator(self.client)
        self.refiner = KnowledgeRefiner(self.evaluator)
        self.rewriter = QueryRewriter(self.client)
        self.searcher = WebSearcher(top_k=top_k_web)
        self.generator = Generator(self.client)

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, query: str) -> CRAGResult:
        """Execute the full CRAG pipeline for *query*."""

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        self._log("Step 1 — Retrieving documents from local corpus …")
        retrieved: list[Document] = self.retriever.retrieve(query, top_k=self.top_k_docs)
        doc_texts = [doc.text for doc in retrieved]
        self._log(f"         Retrieved {len(doc_texts)} document(s).")

        # ── Step 2: Evaluate ──────────────────────────────────────────────────
        self._log("Step 2 — Evaluating retrieval quality …")
        if doc_texts:
            action = self.evaluator.judge(query, doc_texts)
        else:
            action = "INCORRECT"   # nothing retrieved → go to web
        self._log(f"         Action triggered: [{action}]")

        # ── Step 3: Knowledge Correction ──────────────────────────────────────
        internal_knowledge: Optional[str] = None
        external_knowledge: Optional[str] = None

        if action in ("CORRECT", "AMBIGUOUS"):
            self._log("Step 3a — Refining internal knowledge …")
            internal_knowledge = self.refiner.refine(query, doc_texts)
            self._log(f"          Internal knowledge ({len(internal_knowledge)} chars) ready.")

        if action in ("INCORRECT", "AMBIGUOUS"):
            self._log("Step 3b — Rewriting query for web search …")
            search_query = self.rewriter.rewrite(query)
            self._log(f"          Search query: \"{search_query}\"")

            self._log("Step 3b — Searching the web …")
            web_passages = self.searcher.search(search_query)
            self._log(f"          Retrieved {len(web_passages)} web passage(s).")

            if web_passages:
                self._log("Step 3b — Refining external knowledge …")
                external_knowledge = self.refiner.refine(query, web_passages)
                self._log(f"          External knowledge ({len(external_knowledge)} chars) ready.")
            else:
                external_knowledge = ""

        # ── Step 3c: Compose final context ────────────────────────────────────
        if action == "CORRECT":
            final_context = internal_knowledge or ""
        elif action == "INCORRECT":
            final_context = external_knowledge or ""
        else:  # AMBIGUOUS
            parts = filter(None, [internal_knowledge, external_knowledge])
            final_context = " ".join(parts)

        # ── Step 4: Generate ──────────────────────────────────────────────────
        self._log("Step 4 — Generating answer …")
        answer = self.generator.generate(query, final_context)
        self._log("         Done.\n")

        return CRAGResult(
            query=query,
            action=action,
            internal_knowledge=internal_knowledge,
            external_knowledge=external_knowledge,
            final_context=final_context,
            answer=answer,
            retrieved_docs=doc_texts,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CRAG] {msg}")