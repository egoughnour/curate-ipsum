"""
pytest fixtures for Quick Verification testing.

Provides mock backends that work without Docker.
"""
from __future__ import annotations
import json
import pytest
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Mock VerificationBackend
# ---------------------------------------------------------------------------

class MockVerificationResult:
    def __init__(self, status, counterexample=None, stats=None, logs=None):
        self.status = status
        self.counterexample = counterexample
        self.stats = stats or {"mock": True}
        self.logs = logs


class MockVerificationBackend:
    """Configurable mock verification backend for testing."""

    def __init__(self, mode: str = "no_ce"):
        self.mode = mode
        self.call_log: List[Dict[str, Any]] = []

    def verify(self, req: Any) -> MockVerificationResult:
        self.call_log.append({"request": req})
        if self.mode == "ce":
            return MockVerificationResult(
                status="ce_found",
                counterexample={
                    "model": {"x": 337, "y": 1000},
                    "trace": [{"addr": "0x0"}],
                    "path_constraints": [],
                    "notes": {"mock": True},
                },
            )
        if self.mode == "error":
            return MockVerificationResult(status="error", logs="mock error")
        return MockVerificationResult(status="no_ce_within_budget")

    def supports(self) -> dict:
        return {"input": "mock", "constraints": ["any"], "find": ["any"], "avoid": ["any"]}


# ---------------------------------------------------------------------------
# Mock GraphStore
# ---------------------------------------------------------------------------

class MockGraphStore:
    """Minimal mock graph store for testing RAG and orchestrator."""

    def __init__(self):
        self._nodes = {
            "fn_a": {"file_path": "src/a.py", "line_start": 10, "line_end": 20,
                      "symbol_name": "fn_a", "symbol_kind": "function", "partition_id": 0},
            "fn_b": {"file_path": "src/b.py", "line_start": 5, "line_end": 15,
                      "symbol_name": "fn_b", "symbol_kind": "function", "partition_id": 0},
            "fn_c": {"file_path": "src/c.py", "line_start": 1, "line_end": 30,
                      "symbol_name": "fn_c", "symbol_kind": "function", "partition_id": 1},
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
        return [node_id]  # trivial SCC

    def get_node_text(self, node_id: str) -> str:
        return f"def {node_id}():\n    pass\n"

    def get_node_metadata(self, node_id: str) -> Dict[str, Any]:
        return self._nodes.get(node_id, {})

    def get_partition_nodes(self, spec: Any) -> List[str]:
        return list(self._nodes.keys())[:2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_backend_ce():
    return MockVerificationBackend(mode="ce")

@pytest.fixture
def mock_backend_no_ce():
    return MockVerificationBackend(mode="no_ce")

@pytest.fixture
def mock_backend_error():
    return MockVerificationBackend(mode="error")

@pytest.fixture
def mock_graph_store():
    return MockGraphStore()
