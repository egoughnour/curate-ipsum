"""Tests for storage.kuzu_graph_store — Kuzu-backed graph persistence.

These tests require ``kuzu`` to be installed. If it's not available,
all tests are skipped automatically.
"""

import pytest

kuzu = pytest.importorskip("kuzu", reason="kuzu not installed")

from graph.models import (
    CallGraph,
    EdgeKind,
    FunctionSignature,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
)
from storage.kuzu_graph_store import KuzuGraphStore


@pytest.fixture
def store(tmp_path):
    """Create a KuzuGraphStore in a temp directory."""
    db_path = tmp_path / "test_graph.kuzu"
    s = KuzuGraphStore(db_path)
    yield s
    s.close()


@pytest.fixture
def sample_graph():
    """Create a small sample call graph."""
    graph = CallGraph()

    graph.add_node(
        GraphNode(
            id="mod.foo",
            kind=NodeKind.FUNCTION,
            name="foo",
            location=SourceLocation(file="mod.py", line_start=1, line_end=5),
            signature=FunctionSignature(name="foo", params=("x",)),
        )
    )
    graph.add_node(
        GraphNode(
            id="mod.bar",
            kind=NodeKind.FUNCTION,
            name="bar",
            location=SourceLocation(file="mod.py", line_start=7, line_end=12),
        )
    )
    graph.add_node(
        GraphNode(
            id="util.helper",
            kind=NodeKind.FUNCTION,
            name="helper",
            location=SourceLocation(file="util.py", line_start=1, line_end=3),
        )
    )

    graph.add_edge(
        GraphEdge(
            source_id="mod.foo",
            target_id="mod.bar",
            kind=EdgeKind.CALLS,
        )
    )
    graph.add_edge(
        GraphEdge(
            source_id="mod.foo",
            target_id="util.helper",
            kind=EdgeKind.CALLS,
            confidence=0.9,
        )
    )

    return graph


PROJECT_ID = "kuzu-test"


class TestKuzuStoreLoad:
    """Test Kuzu graph store/load round-trip."""

    def test_store_load_roundtrip(self, store, sample_graph):
        """Store and load a graph; verify counts match."""
        store.store_graph(sample_graph, PROJECT_ID)
        loaded = store.load_graph(PROJECT_ID)

        assert loaded is not None
        assert len(loaded.nodes) == 3
        # At least the CALLS edges should be present
        calls = [e for e in loaded.edges if e.kind == EdgeKind.CALLS]
        assert len(calls) == 2

    def test_node_data_preserved(self, store, sample_graph):
        """Verify node attributes survive round-trip."""
        store.store_graph(sample_graph, PROJECT_ID)
        loaded = store.load_graph(PROJECT_ID)

        foo = loaded.get_node("mod.foo")
        assert foo is not None
        assert foo.kind == NodeKind.FUNCTION
        assert foo.name == "foo"

    def test_load_nonexistent_returns_none(self, store):
        """Loading a non-existent project returns None."""
        assert store.load_graph("missing") is None


class TestKuzuNeighbors:
    """Test neighbor queries via Kuzu."""

    def test_outgoing_neighbors(self, store, sample_graph):
        """Get outgoing neighbors."""
        store.store_graph(sample_graph, PROJECT_ID)

        neighbors = store.get_neighbors("mod.foo", PROJECT_ID, direction="outgoing")
        assert "mod.bar" in neighbors
        assert "util.helper" in neighbors

    def test_incoming_neighbors(self, store, sample_graph):
        """Get incoming neighbors."""
        store.store_graph(sample_graph, PROJECT_ID)

        neighbors = store.get_neighbors("mod.bar", PROJECT_ID, direction="incoming")
        assert "mod.foo" in neighbors


class TestKuzuReachability:
    """Test Kameda index persistence in Kuzu."""

    def test_store_and_query_kameda(self, store, sample_graph):
        """Store Kameda labels and query reachability."""
        # Need nodes in the graph for non-planar reach edges
        store.store_graph(sample_graph, PROJECT_ID)

        kameda_data = {
            "left_rank": {"mod.foo": 0, "mod.bar": 1, "util.helper": 2},
            "right_rank": {"mod.foo": 0, "mod.bar": 1, "util.helper": 2},
            "source_id": "virtual_source",
            "sink_id": "virtual_sink",
            "non_planar_reachability": {},
            "all_node_ids": ["mod.foo", "mod.bar", "util.helper"],
        }
        store.store_reachability_index(kameda_data, PROJECT_ID)

        assert store.query_reachable("mod.foo", "mod.bar", PROJECT_ID) is True
        assert store.query_reachable("mod.bar", "mod.foo", PROJECT_ID) is False

    def test_load_kameda_roundtrip(self, store, sample_graph):
        """Store and load Kameda index."""
        store.store_graph(sample_graph, PROJECT_ID)

        kameda_data = {
            "left_rank": {"mod.foo": 0, "mod.bar": 1},
            "right_rank": {"mod.foo": 0, "mod.bar": 1},
            "source_id": "src",
            "sink_id": "sink",
            "non_planar_reachability": {},
            "all_node_ids": ["mod.foo", "mod.bar"],
        }
        store.store_reachability_index(kameda_data, PROJECT_ID)

        loaded = store.load_reachability_index(PROJECT_ID)
        assert loaded is not None
        assert loaded["left_rank"]["mod.foo"] == 0
        assert loaded["source_id"] == "src"


class TestKuzuPartitions:
    """Test partition persistence in Kuzu."""

    def test_store_and_load(self, store, sample_graph):
        """Store and load partitions."""
        store.store_graph(sample_graph, PROJECT_ID)

        partition_tree = {
            "id": "0",
            "depth": 0,
            "fiedler_value": 0.5,
            "node_ids": ["mod.foo", "mod.bar", "util.helper"],
            "children": [
                {
                    "id": "0.0",
                    "depth": 1,
                    "fiedler_value": 0.3,
                    "node_ids": ["mod.foo", "mod.bar"],
                    "children": None,
                },
                {
                    "id": "0.1",
                    "depth": 1,
                    "fiedler_value": 0.4,
                    "node_ids": ["util.helper"],
                    "children": None,
                },
            ],
        }
        store.store_partitions(partition_tree, PROJECT_ID)

        loaded = store.load_partitions(PROJECT_ID)
        assert loaded is not None
        assert loaded["id"] == "0"


class TestKuzuFileHashes:
    """Test file hash operations in Kuzu."""

    def test_set_and_get(self, store):
        """Store and retrieve file hashes."""
        store.set_file_hashes(PROJECT_ID, {"a.py": "hash1", "b.py": "hash2"})

        loaded = store.get_file_hashes(PROJECT_ID)
        assert loaded == {"a.py": "hash1", "b.py": "hash2"}

    def test_empty(self, store):
        """Empty store returns empty dict."""
        assert store.get_file_hashes(PROJECT_ID) == {}


class TestKuzuStats:
    """Test statistics in Kuzu."""

    def test_stats_with_data(self, store, sample_graph):
        """Get stats after storing a graph."""
        store.store_graph(sample_graph, PROJECT_ID)

        stats = store.get_stats(PROJECT_ID)
        assert stats["backend"] == "kuzu"
        assert stats["node_count"] == 3
