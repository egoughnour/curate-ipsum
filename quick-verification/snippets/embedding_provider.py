"""
EmbeddingProvider abstraction — ABC + local sentence-transformers default.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (len(texts), dimension) float32 array of embeddings."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """Uses sentence-transformers with a local model. Default: all-MiniLM-L6-v2 (384-dim)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def dimension(self) -> int:
        return self._dim


class APIEmbeddingProvider(EmbeddingProvider):
    """Delegates to an HTTP embedding API (OpenAI, Cohere, etc.)."""

    def __init__(self, api_url: str, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536):
        self._api_url = api_url
        self._api_key = api_key
        self._model = model
        self._dim = dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        import requests

        resp = requests.post(
            self._api_url,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json={"input": texts, "model": self._model},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return np.array(embeddings, dtype=np.float32)

    def dimension(self) -> int:
        return self._dim
