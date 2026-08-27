"""
retriever/qdrant_store.py

Qdrant Hybrid Search Store using Dense (BAAI/bge-m3) + Sparse (BM25) Embeddings.
Powered by `qdrant-client` and `fastembed`.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

try:
    from qdrant_client import QdrantClient, models
    from fastembed import TextEmbedding, SparseTextEmbedding
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    metadata: dict[str, Any]


class QdrantStore:
    """
    Qdrant vector store configured for Hybrid Search:
    - Dense vectors: BAAI/bge-m3
    - Sparse vectors: Qdrant/bm25
    """

    COLLECTION_NAME = "hardware_dfm_sourcing"
    DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    SPARSE_MODEL_NAME = "Qdrant/bm25"
    SOURCES_FILE = "indexed_sources.json"

    def __init__(
        self,
        store_dir: str = "qdrant_db",
        collection_name: str = COLLECTION_NAME,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client and fastembed are required for QdrantStore. "
                "Install them via `pip install qdrant-client fastembed`."
            )

        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._lock = threading.Lock()

        # Initialize Qdrant Client (local persistent storage by default)
        if host and port:
            self.client = QdrantClient(host=host, port=port)
        else:
            self.client = QdrantClient(path=str(self.store_dir))

        # Lazy loaded embeddings
        self._dense_embedding_model: Optional[TextEmbedding] = None
        self._sparse_embedding_model: Optional[SparseTextEmbedding] = None

        self._indexed_sources: set[str] = set()
        self._source_fingerprints: dict[str, str] = {}
        self._total_chunks_count: int = 0

        self._init_collection()
        self._load_metadata()

    def _get_dense_model(self) -> TextEmbedding:
        if self._dense_embedding_model is None:
            print(f"[QdrantStore] Loading Dense Embedding Model ({self.DENSE_MODEL_NAME})...")
            self._dense_embedding_model = TextEmbedding(model_name=self.DENSE_MODEL_NAME)
        return self._dense_embedding_model

    def _get_sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_embedding_model is None:
            print(f"[QdrantStore] Loading Sparse Embedding Model ({self.SPARSE_MODEL_NAME})...")
            self._sparse_embedding_model = SparseTextEmbedding(model_name=self.SPARSE_MODEL_NAME)
        return self._sparse_embedding_model

    def _init_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            print(f"[QdrantStore] Creating collection '{self.collection_name}' with Hybrid Search config...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1024,  # BAAI/bge-large-en-v1.5 dimension
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )

    def add_chunks(
        self,
        texts: list[str],
        source: str = "unknown",
        source_fingerprint: Optional[str] = None,
        suppress_unchanged_log: bool = True,
        metadata_list: Optional[list[dict[str, Any]]] = None,
    ) -> None:
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
                    print(f"[QdrantStore] '{source}' unchanged — skipping.")
                return

            if already_indexed and source_fingerprint is None:
                if not suppress_unchanged_log:
                    print(f"[QdrantStore] '{source}' already indexed — skipping.")
                return

            is_reindex = already_indexed and not unchanged
            if is_reindex:
                print(f"[QdrantStore] '{source}' changed — removing old points...")
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source",
                                    match=models.MatchValue(value=source),
                                )
                            ]
                        )
                    ),
                )
            else:
                print(f"[QdrantStore] Ingesting {len(texts)} chunk(s) from '{source}'...")

            dense_model = self._get_dense_model()
            sparse_model = self._get_sparse_model()

            dense_vectors = list(dense_model.embed(texts))
            sparse_vectors = list(sparse_model.embed(texts))

            points = []
            start_id = self._get_next_point_id()
            for i, text in enumerate(texts):
                chunk_id = start_id + i
                payload = {
                    "text": text,
                    "source": source,
                    "chunk_id": chunk_id,
                }
                if metadata_list and i < len(metadata_list):
                    payload.update(metadata_list[i])

                dense_vec = dense_vectors[i].tolist()
                sparse_vec = sparse_vectors[i]

                points.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector={
                            "dense": dense_vec,
                            "sparse": models.SparseVector(
                                indices=sparse_vec.indices.tolist(),
                                values=sparse_vec.values.tolist(),
                            ),
                        },
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            self._indexed_sources.add(source)
            if source_fingerprint is not None:
                self._source_fingerprints[source] = source_fingerprint
            self._total_chunks_count = self.client.get_collection(self.collection_name).points_count

            self._save_metadata()

        if is_reindex:
            print(f"[QdrantStore] '{source}' re-indexed. Total chunks: {self._total_chunks_count}")
        else:
            print(f"[QdrantStore] '{source}' indexed. Total chunks: {self._total_chunks_count}")

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Hybrid search combining Dense (BAAI/bge-m3) and Sparse (BM25) via RRF / Query fusion.
        Returns list of (text, score) tuples.
        """
        with self._lock:
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count == 0:
                return []

            dense_model = self._get_dense_model()
            sparse_model = self._get_sparse_model()

            q_dense = list(dense_model.embed([query]))[0].tolist()
            q_sparse = list(sparse_model.embed([query]))[0]

            prefetch = [
                models.Prefetch(
                    query=q_dense,
                    using="dense",
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=q_sparse.indices.tolist(),
                        values=q_sparse.values.tolist(),
                    ),
                    using="sparse",
                    limit=top_k * 2,
                ),
            ]

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
            )

            res = []
            for pt in results.points:
                text = pt.payload.get("text", "")
                score = pt.score
                res.append((text, float(score)))
            return res

    def _get_next_point_id(self) -> int:
        col_info = self.client.get_collection(self.collection_name)
        return col_info.points_count + 1

    def is_indexed(self, source: str) -> bool:
        return source in self._indexed_sources

    def should_ingest(self, source: str, source_fingerprint: Optional[str]) -> bool:
        if source not in self._indexed_sources:
            return True
        if source_fingerprint is None:
            return False
        return self._source_fingerprints.get(source) != source_fingerprint

    def total_chunks(self) -> int:
        col_info = self.client.get_collection(self.collection_name)
        return col_info.points_count

    def indexed_sources(self) -> list[str]:
        return sorted(self._indexed_sources)

    def _save_metadata(self) -> None:
        sources_path = self.store_dir / self.SOURCES_FILE
        with open(sources_path, "w") as f:
            json.dump(
                {
                    "indexed_sources": sorted(self._indexed_sources),
                    "source_fingerprints": self._source_fingerprints,
                },
                f,
                indent=2,
            )

    def _load_metadata(self) -> None:
        sources_path = self.store_dir / self.SOURCES_FILE
        if sources_path.exists():
            try:
                with open(sources_path) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._indexed_sources = set(data.get("indexed_sources", []))
                    self._source_fingerprints = dict(data.get("source_fingerprints", {}))
            except Exception:
                self._indexed_sources = set()
                self._source_fingerprints = {}


# Alias VectorStore to QdrantStore for seamless drop-in compatibility
VectorStore = QdrantStore
