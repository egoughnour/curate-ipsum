"""EmbeddingProvider abstraction — ABC + local sentence-transformers default.

Default model: all-MiniLM-L6-v2 (384-dim, fast, Apache 2.0 licensed).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @abstractmethod
    def embed(self, texts: List[str]) -> list:
        """Return embeddings as list of float-lists, shape (len(texts), dimension)."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.

    Default model: ``all-MiniLM-L6-v2`` (384 dimensions).
    Good balance of speed and quality for code retrieval.

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device (``cpu``, ``cuda``, ``mps``).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._dim: int = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> list:
        """Embed texts using the local model. Returns L2-normalized embeddings."""
        arr = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return arr.tolist()

    def dimension(self) -> int:
        return self._dim
