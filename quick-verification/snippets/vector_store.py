"""
VectorStore abstraction — ABC + FAISS and Qdrant backends.

Mirrors the GraphStore / LLMClient "ABC + swappable backends" pattern.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: Dict[str, Any]


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add(self, ids: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings with IDs and metadata. embeddings shape: (n, dim)."""
        ...

    @abstractmethod
    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Return top_k nearest neighbors. embedding shape: (dim,)."""
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Remove embeddings by ID."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Number of stored embeddings."""
        ...


# ---------------------------------------------------------------------------
# FAISS Backend (zero-dep default)
# ---------------------------------------------------------------------------

class FaissVectorStore(VectorStore):
    """In-process FAISS index with SQLite metadata sidecar."""

    def __init__(self, dimension: int, db_path: str = ":memory:"):
        import faiss
        import sqlite3

        self._dim = dimension
        self._index = faiss.IndexFlatIP(dimension)  # cosine sim on L2-normed vectors
        self._id_map: List[str] = []
        self._meta_map: List[Dict[str, Any]] = []

        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (idx INTEGER PRIMARY KEY, id TEXT, meta TEXT)"
        )
        self._conn.commit()

    def add(self, ids: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        import json
        assert embeddings.shape == (len(ids), self._dim)
        # L2-normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = (embeddings / norms).astype(np.float32)
        base = self._index.ntotal
        self._index.add(normed)
        for i, (eid, meta) in enumerate(zip(ids, metadata)):
            self._id_map.append(eid)
            self._meta_map.append(meta)
            self._conn.execute(
                "INSERT INTO meta (idx, id, meta) VALUES (?, ?, ?)",
                (base + i, eid, json.dumps(meta)),
            )
        self._conn.commit()

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if self._index.ntotal == 0:
            return []
        q = embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._meta_map[idx]
            if filters:
                if not all(meta.get(fk) == fv for fk, fv in filters.items()):
                    continue
            results.append(SearchResult(id=self._id_map[idx], score=float(score), metadata=meta))
        return results

    def delete(self, ids: List[str]) -> None:
        # FAISS IndexFlatIP doesn't support removal; rebuild needed
        # For dev use, this is acceptable
        raise NotImplementedError("FAISS flat index does not support deletion. Rebuild the index.")

    def count(self) -> int:
        return self._index.ntotal


# ---------------------------------------------------------------------------
# Qdrant Backend (containerized, production)
# ---------------------------------------------------------------------------

class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store. Requires a running Qdrant instance."""

    def __init__(self, url: str = "http://localhost:6333", collection: str = "code_nodes", dimension: int = 384):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(url=url)
        self._collection = collection
        self._dim = dimension

        # Ensure collection exists
        collections = [c.name for c in self._client.get_collections().collections]
        if collection not in collections:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def add(self, ids: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        from qdrant_client.models import PointStruct
        import hashlib

        points = []
        for eid, emb, meta in zip(ids, embeddings, metadata):
            # Qdrant needs int or UUID point IDs; hash string ID to int
            pid = int(hashlib.sha256(eid.encode()).hexdigest()[:15], 16)
            points.append(PointStruct(id=pid, vector=emb.tolist(), payload={"node_id": eid, **meta}))
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qfilter = None
        if filters:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            qfilter = Filter(must=conditions)

        hits = self._client.search(
            collection_name=self._collection,
            query_vector=embedding.tolist(),
            limit=top_k,
            query_filter=qfilter,
        )
        return [
            SearchResult(
                id=h.payload.get("node_id", str(h.id)),
                score=h.score,
                metadata={k: v for k, v in h.payload.items() if k != "node_id"},
            )
            for h in hits
        ]

    def delete(self, ids: List[str]) -> None:
        import hashlib
        pids = [int(hashlib.sha256(eid.encode()).hexdigest()[:15], 16) for eid in ids]
        self._client.delete(collection_name=self._collection, points_selector=pids)

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count
