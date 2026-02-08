"""VectorStore abstraction — ABC + Chroma backend.

Chroma was selected as the default vector store because it is:
  - Nearly zero-dep (pip install chromadb)
  - Supports persistent storage out of the box
  - Embeddable (in-process) OR client/server mode
  - Handles metadata filtering natively
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """A single vector search result."""
    id: str
    score: float
    metadata: Dict[str, Any]


class VectorStore(ABC):
    """Abstract vector store interface.

    Implementations must support:
      - Adding embeddings with IDs and metadata
      - Nearest-neighbor search with optional metadata filters
      - Deletion by ID
      - Count of stored embeddings
    """

    @abstractmethod
    def add(self, ids: List[str], embeddings: list, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings. ``embeddings`` is a list of float-lists, shape (n, dim)."""
        ...

    @abstractmethod
    def search(
        self,
        embedding: list,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Return top_k nearest neighbors. ``embedding`` is a single float-list."""
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Remove embeddings by ID."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Number of stored embeddings."""
        ...


class ChromaVectorStore(VectorStore):
    """Chroma-backed vector store.

    Supports two operational modes:

    1. **Embedded (default)**: In-process with persistent storage.
       No external service needed — just a directory on disk.

       >>> store = ChromaVectorStore(persist_dir="./data/chroma")

    2. **Client/server**: Connects to a running Chroma server (Docker).

       >>> store = ChromaVectorStore(host="localhost", port=8000)

    Args:
        collection_name: Name of the Chroma collection.
        persist_dir: Directory for embedded persistent storage (mode 1).
        host: Chroma server host (mode 2). Overrides persist_dir.
        port: Chroma server port (mode 2).
    """

    def __init__(
        self,
        collection_name: str = "code_nodes",
        persist_dir: Optional[str] = "./data/chroma",
        host: Optional[str] = None,
        port: int = 8000,
    ):
        import chromadb

        if host:
            # Client/server mode
            self._client = chromadb.HttpClient(host=host, port=port)
        else:
            # Embedded persistent mode
            self._client = chromadb.PersistentClient(path=persist_dir or "./data/chroma")

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: List[str], embeddings: list, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings with IDs and metadata.

        Chroma handles upsert natively — duplicate IDs are updated.
        """
        # Chroma expects embeddings as list of lists of floats
        emb_lists = [list(map(float, e)) for e in embeddings]

        # Sanitize metadata: Chroma requires string/int/float/bool values only
        clean_meta = [self._sanitize_metadata(m) for m in metadata]

        self._collection.upsert(
            ids=ids,
            embeddings=emb_lists,
            metadatas=clean_meta,
        )

    def search(
        self,
        embedding: list,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for nearest neighbors.

        ``filters`` is translated to a Chroma ``where`` clause.
        Example: ``{"partition_id": 3}`` → ``{"partition_id": {"$eq": 3}}``
        """
        query_embedding = [list(map(float, embedding))]

        where = None
        if filters:
            where = {k: {"$eq": v} for k, v in filters.items()}
            # Chroma requires $and for multiple conditions
            if len(where) > 1:
                where = {"$and": [{k: v} for k, v in where.items()]}

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, max(self._collection.count(), 1)),
            where=where,
        )

        out: List[SearchResult] = []
        if results and results["ids"]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
            for rid, dist, meta in zip(ids, distances, metadatas):
                # Chroma returns distance; convert to similarity score
                score = 1.0 - dist  # cosine distance → cosine similarity
                out.append(SearchResult(id=rid, score=score, metadata=meta or {}))
        return out

    def delete(self, ids: List[str]) -> None:
        """Delete by IDs."""
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata values are Chroma-compatible (str/int/float/bool)."""
        clean = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif v is None:
                continue
            else:
                clean[k] = str(v)
        return clean
