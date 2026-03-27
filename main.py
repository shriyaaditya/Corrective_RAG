"""
main.py

Interactive CRAG demo with automatic document ingestion.

How it works
------------
1. On startup, all files in the `docs/` folder are parsed and indexed
   into a local FAISS vector store (stored in `vector_db/`).
2. A background watcher monitors `docs/` — drop any new PDF, DOCX, XLSX,
   TXT, CSV, or MD file in and it is automatically indexed within seconds.
3. You can ask questions in the interactive prompt; CRAG retrieves
   relevant chunks from your documents, evaluates quality, optionally
   searches the web, and generates a grounded answer.

Usage
-----
    python main.py                        # interactive
    python main.py --query "your question"
    python main.py --docs-dir my_docs --db-dir my_vector_db
    python main.py --no-verbose
    python main.py --list-sources
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from retriever.document_parser import DocumentParser
from retriever.document_watcher import DocumentWatcher
from retriever.vector_store import VectorStore
from retriever.vector_retriever import VectorRetriever
from crag_pipeline import CRAGPipeline


def _print_result(result) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Query  : {result.query}")
    print(f"  Action : {result.action}")
    print(f"{sep}")
    print("  Answer :")
    print(textwrap.fill(result.answer, width=72, initial_indent="    ", subsequent_indent="    "))
    print(sep)
    if result.internal_knowledge:
        preview = result.internal_knowledge[:400]
        if len(result.internal_knowledge) > 400:
            preview += "…"
        print("\n  [Internal Knowledge Used]")
        print(textwrap.fill(preview, width=72, initial_indent="    ", subsequent_indent="    "))
    if result.external_knowledge:
        preview = result.external_knowledge[:400]
        if len(result.external_knowledge) > 400:
            preview += "…"
        print("\n  [External Knowledge Used (web)]")
        print(textwrap.fill(preview, width=72, initial_indent="    ", subsequent_indent="    "))
    print()


def _print_sources(store: VectorStore) -> None:
    sources = store.indexed_sources()
    if not sources:
        print("\n[main] No documents indexed yet.")
        print(f"       Drop files into the docs/ folder to index them.\n")
        return
    print(f"\n[main] {len(sources)} indexed source(s):")
    for s in sources:
        print(f"       - {s}")
    print(f"       Total chunks: {store.total_chunks()}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="CRAG — Corrective Retrieval Augmented Generation")
    ap.add_argument("--docs-dir", default="docs", help="Folder to watch for documents (default: docs/)")
    ap.add_argument("--db-dir",   default="vector_db", help="FAISS index folder (default: vector_db/)")
    ap.add_argument("--query",    type=str, default=None, help="Single query, then exit")
    ap.add_argument("--no-verbose", action="store_true", help="Suppress pipeline logs")
    ap.add_argument("--list-sources", action="store_true", help="Print indexed sources and exit")
    args = ap.parse_args()

    print("\n=== CRAG System Starting ===\n")

    store   = VectorStore(store_dir=args.db_dir)
    parser_ = DocumentParser()
    watcher = DocumentWatcher(docs_dir=args.docs_dir, store=store, parser=parser_)
    watcher.start()

    if args.list_sources:
        _print_sources(store)
        watcher.stop()
        return

    retriever = VectorRetriever(store)
    pipeline  = CRAGPipeline(corpus_retriever=retriever, verbose=not args.no_verbose)

    if store.total_chunks() == 0:
        print(
            "\n[warning] No documents indexed yet.\n"
            f"   Drop files into '{args.docs_dir}/' — they index automatically.\n"
            "   Queries will fall back to web search until then.\n"
        )

    if args.query:
        result = pipeline.run(args.query)
        _print_result(result)
        watcher.stop()
        return

    print("\nCRAG Interactive Mode")
    print("    :sources  — list indexed documents")
    print("    :help     — show commands")
    print("    exit      — quit\n")

    while True:
        try:
            raw = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if raw.lower() == ":sources":
            _print_sources(store)
            continue
        if raw.lower() == ":help":
            print(
                "\n  :sources  — list indexed documents\n"
                "  :help     — this message\n"
                "  exit      — quit\n\n"
                f"  Drop PDF/DOCX/XLSX/TXT/CSV/MD into '{args.docs_dir}/' to index.\n"
            )
            continue

        result = pipeline.run(raw)
        _print_result(result)

    watcher.stop()


if __name__ == "__main__":
    main()