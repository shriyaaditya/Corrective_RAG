"""
retriever/vector_store.py

Persistent vector store built on FAISS + sentence-transformers.

Everything is FREE and runs locally:
  - sentence-transformers downloads "all-MiniLM-L6-v2" once (~90 MB) to
    ~/.cache/huggingface and reuses it on every subsequent run.
  - FAISS runs entirely in-process on CPU.

The store saves itself to disk after every batch add so nothing is lost
between sessions.

Usage
-----
    store = VectorStore(store_dir="vector_db")
    store.add_chunks(["text one", "text two"], source="report.pdf")

    results = store.search("what is RAG?", top_k=5)
    for chunk, score in results:
        print(score, chunk[:80])
"""

from __future__ import annotations

import json
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str          # filename that produced this chunk
    chunk_id: int        # index within that file


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thread-safe FAISS vector store with disk persistence.

    Parameters
    ----------
    store_dir : str
        Directory where the FAISS index and metadata are saved.
        Created automatically if it doesn't exist.
    model_name : str
        Any sentence-transformers model name.
        "all-MiniLM-L6-v2" is fast, small (~90 MB), and good quality.
    """

    INDEX_FILE    = "faiss.index"
    META_FILE     = "metadata.pkl"
    SOURCES_FILE  = "indexed_sources.json"

    def __init__(
        self,
        store_dir: str = "vector_db",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.store_dir  = Path(store_dir)
        self.model_name = model_name
        self._lock      = threading.Lock()

        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load heavy libs so import is fast
        self._model  = None
        self._index  = None
        self._chunks: list[Chunk] = []
        # Track which source files have already been indexed
        self._indexed_sources: set[str] = set()
        # Track last fingerprint (e.g. mtime/hash) for each source file
        self._source_fingerprints: dict[str, str] = {}

        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        texts: list[str],
        source: str = "unknown",
        source_fingerprint: Optional[str] = None,
        suppress_unchanged_log: bool = True,
    ) -> None:
        """
        Embed *texts* and add them to the FAISS index.
        If *source* already exists:
          - same fingerprint  -> skip (unchanged)
          - different fingerprint -> replace old chunks and re-index
        """
        if not texts:
            return

        with self._lock:
            prev_fp = self._source_fingerprints.get(source)
            already_indexed = source in self._indexed_sources
            unchanged = (
                already_indexed
                and source_fingerprint is not None
                and prev_fp == source_fingerprint
            )

            if unchanged:
                if not suppress_unchanged_log:
                    print(f"[VectorStore] '{source}' unchanged — skipping.")
                return

            if already_indexed and source_fingerprint is None:
                if not suppress_unchanged_log:
                    print(f"[VectorStore] '{source}' already indexed — skipping.")
                return

            is_reindex = already_indexed and not unchanged
            if is_reindex:
                print(f"[VectorStore] '{source}' changed — re-indexing …")
                self._chunks = [c for c in self._chunks if c.source != source]
            else:
                print(f"[VectorStore] Embedding {len(texts)} chunk(s) from '{source}' …")

            start_id = len(self._chunks)
            for i, text in enumerate(texts):
                self._chunks.append(Chunk(text=text, source=source, chunk_id=start_id + i))

            self._indexed_sources.add(source)
            if source_fingerprint is not None:
                self._source_fingerprints[source] = source_fingerprint

            self._rebuild_index_locked()

        self._save()
        if is_reindex:
            print(f"[VectorStore] '{source}' re-indexed. Total chunks: {len(self._chunks)}")
        else:
            print(f"[VectorStore] '{source}' indexed. Total chunks: {len(self._chunks)}")

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return the top-k most relevant (text, score) pairs for *query*.
        Score is cosine similarity in [0, 1]; higher = more relevant.
        Returns [] if the store is empty.
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []

            import faiss
            import numpy as np

            q_emb = self._embed([query])
            faiss.normalize_L2(q_emb)

            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(q_emb, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((self._chunks[idx].text, float(score)))
            return results

    def is_indexed(self, source: str) -> bool:
        return source in self._indexed_sources

    def should_ingest(self, source: str, source_fingerprint: Optional[str]) -> bool:
        """
        True when source is new OR changed since last indexed version.
        """
        if source not in self._indexed_sources:
            return True
        if source_fingerprint is None:
            return False
        return self._source_fingerprints.get(source) != source_fingerprint

    def total_chunks(self) -> int:
        return len(self._chunks)

    def indexed_sources(self) -> list[str]:
        return sorted(self._indexed_sources)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        import faiss
        with self._lock:
            if self._index is not None:
                faiss.write_index(self._index, str(self.store_dir / self.INDEX_FILE))
            with open(self.store_dir / self.META_FILE, "wb") as f:
                pickle.dump(self._chunks, f)
            with open(self.store_dir / self.SOURCES_FILE, "w") as f:
                json.dump(
                    {
                        "indexed_sources": sorted(self._indexed_sources),
                        "source_fingerprints": self._source_fingerprints,
                    },
                    f,
                    indent=2,
                )

    def _load(self) -> None:
        index_path   = self.store_dir / self.INDEX_FILE
        meta_path    = self.store_dir / self.META_FILE
        sources_path = self.store_dir / self.SOURCES_FILE
        index_loaded = False

        if index_path.exists() and meta_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(index_path))
                with open(meta_path, "rb") as f:
                    self._chunks = pickle.load(f)
                index_loaded = True
                print(f"[VectorStore] Loaded existing index — {len(self._chunks)} chunk(s).")
            except Exception as e:
                print(f"[VectorStore] Could not load existing index: {e}. Starting fresh.")
                self._index  = None
                self._chunks = []
                self._indexed_sources = set()
                self._source_fingerprints = {}

        if sources_path.exists() and (index_loaded or (not index_path.exists() and not meta_path.exists())):
            try:
                with open(sources_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    # Backward compatibility with older format
                    self._indexed_sources = set(data)
                    self._source_fingerprints = {}
                elif isinstance(data, dict):
                    self._indexed_sources = set(data.get("indexed_sources", []))
                    self._source_fingerprints = dict(data.get("source_fingerprints", {}))
                else:
                    self._indexed_sources = set()
                    self._source_fingerprints = {}
            except Exception:
                self._indexed_sources = set()
                self._source_fingerprints = {}

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]):
        """Embed a list of strings → float32 numpy array (n, dim)."""
        import numpy as np

        if self._model is None:
            print(f"[VectorStore] Loading embedding model '{self.model_name}' …")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            print(f"[VectorStore] Model loaded.")

        vecs = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,   # we normalise manually for FAISS
        )
        return vecs.astype("float32")

    def _rebuild_index_locked(self) -> None:
        """
        Rebuild the FAISS index from current chunks.
        Must be called with self._lock held.
        """
        import faiss

        if not self._chunks:
            self._index = None
            return

        texts = [c.text for c in self._chunks]
        embeddings = self._embed(texts)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self._index = index