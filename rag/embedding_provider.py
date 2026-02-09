"""
Embedding provider abstraction with local sentence-transformers backend.

sentence-transformers and all-MiniLM-L6-v2 are core dependencies (not optional).
Alternative/larger models can be installed via [embeddings-gpu] or [embeddings-large].

Decision: D-017
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

LOG = logging.getLogger("rag.embedding_provider")


class EmbeddingProvider(ABC):
    """Abstract interface for text → embedding vector conversion."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a batch of texts into embedding vectors.

        Returns a list of float vectors, one per input text.
        """
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding via sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good for code).
    Install [embeddings-gpu] for GPU acceleration or [embeddings-large]
    for InstructorEmbedding support.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        LOG.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    def dimension(self) -> int:
        return self._dim


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing. Returns fixed-length zero vectors."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]

    def dimension(self) -> int:
        return self._dim
