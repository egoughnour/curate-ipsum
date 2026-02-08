"""Shared pytest fixtures for Quick Verification tests."""
from __future__ import annotations

import pytest
from typing import Any, Dict, List

# Import from the package
from quickverify.verification.types import (
    Budget,
    Counterexample,
    Predicate,
    SymbolSpec,
    VerificationRequest,
    VerificationResult,
    VerificationTarget,
)
from quickverify.verification.backends.mock import MockBackend


# -- Mock GraphStore for RAG testing --

class MockGraphStore:
    """Minimal mock graph store."""

    def __init__(self):
        self._nodes = {
            "fn_a": {
                "file_path": "src/a.py", "line_start": 10, "line_end": 20,
                "symbol_name": "fn_a", "symbol_kind": "function", "partition_id": 0,
            },
            "fn_b": {
                "file_path": "src/b.py", "line_start": 5, "line_end": 15,
                "symbol_name": "fn_b", "symbol_kind": "function", "partition_id": 0,
            },
            "fn_c": {
                "file_path": "src/c.py", "line_start": 1, "line_end": 30,
                "symbol_name": "fn_c", "symbol_kind": "function", "partition_id": 1,
            },
        }
        self._calls = {"fn_a": ["fn_b"], "fn_b": ["fn_c"], "fn_c": []}
        self._callers = {"fn_a": [], "fn_b": ["fn_a"], "fn_c": ["fn_b"]}

    def get_callers(self, node_id: str) -> List[str]:
        return self._callers.get(node_id, [])

    def get_callees(self, node_id: str) -> List[str]:
        return self._calls.get(node_id, [])

    def get_partition_members(self, partition_id: int) -> List[str]:
        return [nid for nid, m in self._nodes.items() if m.get("partition_id") == partition_id]

    def get_scc_members(self, node_id: str) -> List[str]:
        return [node_id]

    def get_node_text(self, node_id: str) -> str:
        return f"def {node_id}():\n    pass\n"

    def get_node_metadata(self, node_id: str) -> Dict[str, Any]:
        return self._nodes.get(node_id, {})

    def get_partition_nodes(self, spec: Any) -> List[str]:
        return list(self._nodes.keys())[:2]


# -- Mock VectorStore --

class MockVectorStore:
    """Minimal mock vector store for RAG testing."""

    def __init__(self):
        self._data: Dict[str, dict] = {}

    def add(self, ids, embeddings, metadata):
        for i, eid in enumerate(ids):
            self._data[eid] = {"embedding": embeddings[i], "metadata": metadata[i]}

    def search(self, embedding, top_k=10, filters=None):
        from quickverify.rag.vector_store import SearchResult
        results = []
        for eid, data in list(self._data.items())[:top_k]:
            results.append(SearchResult(id=eid, score=0.9, metadata=data["metadata"]))
        return results

    def delete(self, ids):
        for eid in ids:
            self._data.pop(eid, None)

    def count(self):
        return len(self._data)


# -- Mock EmbeddingProvider --

class MockEmbeddingProvider:
    """Returns fixed-dimension zero vectors."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed(self, texts):
        return [[0.0] * self._dim for _ in texts]

    def dimension(self):
        return self._dim


# -- Fixtures --

@pytest.fixture
def mock_backend_ce():
    return MockBackend(mode="ce")

@pytest.fixture
def mock_backend_no_ce():
    return MockBackend(mode="no_ce")

@pytest.fixture
def mock_backend_error():
    return MockBackend(mode="error")

@pytest.fixture
def mock_graph_store():
    return MockGraphStore()

@pytest.fixture
def mock_vector_store():
    return MockVectorStore()

@pytest.fixture
def mock_embedding_provider():
    return MockEmbeddingProvider()

@pytest.fixture
def sample_request():
    return VerificationRequest(
        target=VerificationTarget(binary_name="harness", entry="target_fn"),
        symbols=[
            SymbolSpec(name="x", bits=32, kind="int"),
            SymbolSpec(name="y", bits=32, kind="int"),
        ],
        constraints=["x>=0", "x<=2000", "y>=0", "y<=2000"],
        find=Predicate(kind="addr_reached", value="violation"),
        avoid=Predicate(kind="addr_avoided", value="ok_exit"),
        budget=Budget(timeout_s=10, max_states=256, max_path_len=200, max_loop_iters=8),
        metadata={"test": True},
    )
