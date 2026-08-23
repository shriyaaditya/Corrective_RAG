"""
scripts/test_routing_paths.py

Verification script to test the 3 CRAG routing paths:
1. CORRECT Path (Static engineering/manufacturing specs)
2. AMBIGUOUS Path (Volatile component stock/pricing/MPN queries requiring market API)
3. INCORRECT/Refusal Path (Out-of-domain or ungrounded queries)
"""

import sys
from pathlib import Path

# Add project root and app directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

from app.services.retriever.qdrant_store import QdrantStore
from app.services.retriever.vector_retriever import VectorRetriever
from app.services.retriever.document_parser import DocumentParser
from app.services.crag_pipeline import CRAGPipeline


def index_data_folder(store: QdrantStore):
    """Index all files in data/ directory into Qdrant."""
    parser = DocumentParser()
    data_dir = root_dir / "data"
    if not data_dir.exists():
        print(f"[Watcher] Directory {data_dir} does not exist.")
        return

    files = [p for p in data_dir.iterdir() if p.is_file() and parser.is_supported(str(p))]
    print(f"\n[Indexing] Found {len(files)} file(s) in data/ directory for Qdrant indexing...")

    for f in files:
        chunks = parser.parse(str(f))
        store.add_chunks(
            texts=chunks,
            source=f.name,
            source_fingerprint="data_v1"
        )
    print(f"[Indexing] Completed. Total vector points: {store.total_chunks()}\n")


def main():
    store = QdrantStore(store_dir=str(root_dir / "qdrant_db"))
    index_data_folder(store)

    retriever = VectorRetriever(store)
    pipeline = CRAGPipeline(corpus_retriever=retriever, verbose=True)

    queries = [
        ("Query 1 (The CORRECT Path)", "What is the maximum ramp-up rate and the required peak temperature range for a SAC305 alloy reflow profile?"),
        ("Query 2 (The AMBIGUOUS / Fallback Path)", "What is the global stock, unit price, and manufacturer part number for internal part number IPN-CSA-003?"),
        ("Query 3 (The INCORRECT / Hallucination Path)", "What is the specific solder reflow temperature profile for a BGA-256 package in a quantum computing array?"),
    ]

    for label, query in queries:
        print("\n" + "="*80)
        print(f"RUNNING TEST: {label}")
        print(f"QUERY: {query}")
        print("="*80)

        result = pipeline.run(query)

        print("\n" + "-"*60)
        print(f"RESULT TRACE FOR [{label}]")
        print(f"QUERY  : {result.query}")
        print(f"ACTION : {result.action}")
        print(f"ANSWER :\n{result.answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()
