"""
Abstract vector store interface with Chroma backend.

chromadb is a core dependency. Chroma operates in two modes:
- Embedded PersistentClient: no server needed, local persistence
- HttpClient: connects to the Chroma Docker service (docker compose up -d)

Follows the D-014 pattern (abstract base + factory function).
Decision: D-017
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("rag.vector_store")


@dataclass
class VectorDocument:
    """A document stored in the vector store."""

    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """A single search result from the vector store."""

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """
    Abstract interface for vector storage and similarity search.

    Implementations persist document embeddings and support
    approximate nearest-neighbor queries.
    """

    @abstractmethod
    def add(
        self,
        documents: List[VectorDocument],
    ) -> None:
        """Add documents to the store. Upserts on matching IDs."""

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar documents by embedding vector."""

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""

    def close(self) -> None:
        """Release resources. Override if needed."""
        pass


class ChromaVectorStore(VectorStore):
    """
    Chroma-based vector store (core dependency).

    Operates in two modes:
    - Embedded (PersistentClient): no server, local persistence
    - Client/server (HttpClient): connects to docker compose chroma service

    Set CHROMA_HOST env var to connect to a remote Chroma instance,
    otherwise defaults to embedded mode.

    Decision: D-017
    """

    def __init__(
        self,
        collection_name: str = "code_nodes",
        persist_directory: Optional[str] = None,
        chroma_host: Optional[str] = None,
        chroma_port: int = 8000,
    ) -> None:
        import chromadb

        if chroma_host:
            self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            LOG.info("Chroma: connected to %s:%d", chroma_host, chroma_port)
        elif persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_directory)
            LOG.info("Chroma: persistent at %s", persist_directory)
        else:
            self._client = chromadb.Client()
            LOG.info("Chroma: ephemeral (in-memory)")

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: List[VectorDocument]) -> None:
        if not documents:
            return

        ids = [d.id for d in documents]
        texts = [d.text for d in documents]
        metadatas = [d.metadata if d.metadata else {"_placeholder": "true"} for d in documents]

        kwargs: Dict[str, Any] = {
            "ids": ids,
            "documents": texts,
            "metadatas": metadatas,
        }

        # Include embeddings if provided
        embeddings = [d.embedding for d in documents if d.embedding is not None]
        if len(embeddings) == len(documents):
            kwargs["embeddings"] = embeddings

        self._collection.upsert(**kwargs)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        n = min(top_k, self._collection.count())
        if n == 0:
            return []

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n,
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self._collection.query(**kwargs)

        out: List[VectorSearchResult] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                score = max(0.0, 1.0 - distance)  # cosine distance → similarity
                text = results["documents"][0][i] if results.get("documents") else ""
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                out.append(VectorSearchResult(id=doc_id, text=text, score=score, metadata=meta))

        return out

    def delete(self, ids: List[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()


def build_vector_store(
    backend: str = "chroma",
    **kwargs: Any,
) -> VectorStore:
    """
    Factory: create a VectorStore of the requested type.

    Args:
        backend: "chroma" (only supported backend currently)
        **kwargs: Backend-specific configuration

    Returns:
        VectorStore instance

    Raises:
        ValueError: Unknown backend
    """
    if backend == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(
            f"Unknown vector store backend: {backend!r}. "
            f"Supported: 'chroma'"
        )
