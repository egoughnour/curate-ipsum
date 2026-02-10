"""Tests for storage.sqlite_graph_store — SQLite-backed graph persistence."""

import pytest

from curate_ipsum.graph.models import (
    CallGraph,
    EdgeKind,
    FunctionSignature,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
)
from curate_ipsum.storage.sqlite_graph_store import SQLiteGraphStore


@pytest.fixture
def store(tmp_path):
    """Create a SQLiteGraphStore in a temp directory."""
    db_path = tmp_path / "test_graph.db"
    s = SQLiteGraphStore(db_path)
    yield s
    s.close()


@pytest.fixture
def sample_graph():
    """Create a small sample call graph."""
    graph = CallGraph()

    # Add modules and functions
    graph.add_node(
        GraphNode(
            id="mod.foo",
            kind=NodeKind.FUNCTION,
            name="foo",
            location=SourceLocation(file="mod.py", line_start=1, line_end=5),
            signature=FunctionSignature(name="foo", params=("x", "y"), return_type="int"),
            docstring="Compute foo.",
        )
    )
    graph.add_node(
        GraphNode(
            id="mod.bar",
            kind=NodeKind.FUNCTION,
            name="bar",
            location=SourceLocation(file="mod.py", line_start=7, line_end=12),
            signature=FunctionSignature(name="bar", params=("z",)),
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

    # Edges
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
            is_conditional=True,
        )
    )
    graph.add_edge(
        GraphEdge(
            source_id="mod.bar",
            target_id="util.helper",
            kind=EdgeKind.REFERENCES,
        )
    )

    return graph


PROJECT_ID = "test-project"


class TestStoreLoadGraph:
    """Test graph store/load round-trip."""

    def test_store_load_roundtrip(self, store, sample_graph):
        """Store and load a graph; verify node/edge counts match."""
        store.store_graph(sample_graph, PROJECT_ID)
        loaded = store.load_graph(PROJECT_ID)

        assert loaded is not None
        assert len(loaded.nodes) == 3
        assert len(loaded.edges) == 3

    def test_node_data_preserved(self, store, sample_graph):
        """Verify node attributes survive round-trip."""
        store.store_graph(sample_graph, PROJECT_ID)
        loaded = store.load_graph(PROJECT_ID)

        foo = loaded.get_node("mod.foo")
        assert foo is not None
        assert foo.kind == NodeKind.FUNCTION
        assert foo.name == "foo"
        assert foo.location.file == "mod.py"
        assert foo.location.line_start == 1
        assert foo.signature.params == ("x", "y")
        assert foo.signature.return_type == "int"
        assert foo.docstring == "Compute foo."

    def test_edge_data_preserved(self, store, sample_graph):
        """Verify edge attributes survive round-trip."""
        store.store_graph(sample_graph, PROJECT_ID)
        loaded = store.load_graph(PROJECT_ID)

        calls_edges = list(loaded.get_edges_from("mod.foo", EdgeKind.CALLS))
        assert len(calls_edges) == 2

        # Find the conditional edge
        cond_edges = [e for e in calls_edges if e.is_conditional]
        assert len(cond_edges) == 1
        assert cond_edges[0].confidence == 0.9
        assert cond_edges[0].target_id == "util.helper"

    def test_load_nonexistent_project(self, store):
        """Loading a non-existent project returns None."""
        assert store.load_graph("missing-project") is None

    def test_overwrite_existing(self, store, sample_graph):
        """Storing a new graph for the same project replaces the old one."""
        store.store_graph(sample_graph, PROJECT_ID)

        # Store a smaller graph
        small = CallGraph()
        small.add_node(GraphNode(id="only.one", kind=NodeKind.FUNCTION, name="one"))
        store.store_graph(small, PROJECT_ID)

        loaded = store.load_graph(PROJECT_ID)
        assert len(loaded.nodes) == 1
        assert "only.one" in loaded.nodes

    def test_schema_idempotent(self, tmp_path):
        """Opening the same DB twice doesn't error."""
        db_path = tmp_path / "graph.db"
        s1 = SQLiteGraphStore(db_path)
        s2 = SQLiteGraphStore(db_path)  # Same DB
        s1.close()
        s2.close()


class TestSingleNodeEdge:
    """Test single node/edge operations."""

    def test_store_and_get_node(self, store):
        """Store a single node and retrieve it."""
        store.store_node(
            {
                "id": "my.func",
                "kind": "function",
                "name": "func",
                "file_path": "my.py",
                "line_start": 10,
                "line_end": 20,
            },
            PROJECT_ID,
        )

        result = store.get_node("my.func", PROJECT_ID)
        assert result is not None
        assert result["id"] == "my.func"
        assert result["kind"] == "function"
        assert result["file_path"] == "my.py"

    def test_get_nonexistent_node(self, store):
        """Getting a non-existent node returns None."""
        assert store.get_node("missing", PROJECT_ID) is None


class TestNeighbors:
    """Test neighbor queries."""

    def test_outgoing_neighbors(self, store, sample_graph):
        """Get outgoing neighbors (callees)."""
        store.store_graph(sample_graph, PROJECT_ID)

        neighbors = store.get_neighbors("mod.foo", PROJECT_ID, direction="outgoing")
        assert set(neighbors) == {"mod.bar", "util.helper"}

    def test_incoming_neighbors(self, store, sample_graph):
        """Get incoming neighbors (callers)."""
        store.store_graph(sample_graph, PROJECT_ID)

        neighbors = store.get_neighbors("util.helper", PROJECT_ID, direction="incoming")
        assert "mod.foo" in neighbors

    def test_both_directions(self, store, sample_graph):
        """Get neighbors in both directions."""
        store.store_graph(sample_graph, PROJECT_ID)

        neighbors = store.get_neighbors("mod.bar", PROJECT_ID, direction="both")
        # Incoming: mod.foo (calls), outgoing: util.helper (references)
        assert "mod.foo" in neighbors
        assert "util.helper" in neighbors

    def test_filter_by_edge_kind(self, store, sample_graph):
        """Filter neighbors by edge kind."""
        store.store_graph(sample_graph, PROJECT_ID)

        calls_only = store.get_neighbors("mod.foo", PROJECT_ID, direction="outgoing", edge_kind="calls")
        assert set(calls_only) == {"mod.bar", "util.helper"}

        ref_only = store.get_neighbors("mod.bar", PROJECT_ID, direction="outgoing", edge_kind="references")
        assert ref_only == ["util.helper"]


class TestReachabilityIndex:
    """Test Kameda index persistence."""

    def test_store_and_query_kameda(self, store):
        """Store Kameda labels and query reachability."""
        kameda_data = {
            "left_rank": {"A": 0, "B": 1, "C": 2},
            "right_rank": {"A": 0, "B": 1, "C": 2},
            "source_id": "virtual_source",
            "sink_id": "virtual_sink",
            "non_planar_reachability": {},
            "all_node_ids": ["A", "B", "C"],
        }
        store.store_reachability_index(kameda_data, PROJECT_ID)

        # A reaches B and C (A's labels <= B's and C's labels)
        assert store.query_reachable("A", "B", PROJECT_ID) is True
        assert store.query_reachable("A", "C", PROJECT_ID) is True

        # C does not reach A
        assert store.query_reachable("C", "A", PROJECT_ID) is False

    def test_nonplanar_fallback(self, store):
        """Non-planar reachability table is used as fallback."""
        kameda_data = {
            "left_rank": {"X": 5, "Y": 0},
            "right_rank": {"X": 5, "Y": 0},
            "source_id": "s",
            "sink_id": "t",
            "non_planar_reachability": {"X": ["Y"]},  # X reaches Y via NP edge
            "all_node_ids": ["X", "Y"],
        }
        store.store_reachability_index(kameda_data, PROJECT_ID)

        # X reaches Y via non-planar fallback even though Kameda says no
        assert store.query_reachable("X", "Y", PROJECT_ID) is True

    def test_load_kameda_roundtrip(self, store):
        """Store and load Kameda index."""
        kameda_data = {
            "left_rank": {"n1": 0, "n2": 1},
            "right_rank": {"n1": 0, "n2": 1},
            "source_id": "src",
            "sink_id": "sink",
            "non_planar_reachability": {"n1": ["n2"]},
            "all_node_ids": ["n1", "n2"],
        }
        store.store_reachability_index(kameda_data, PROJECT_ID)

        loaded = store.load_reachability_index(PROJECT_ID)
        assert loaded is not None
        assert loaded["left_rank"] == {"n1": 0, "n2": 1}
        assert loaded["source_id"] == "src"
        assert "n2" in loaded["non_planar_reachability"].get("n1", set())

    def test_no_kameda_returns_none(self, store):
        """Loading from empty store returns None."""
        assert store.load_reachability_index(PROJECT_ID) is None


class TestPartitions:
    """Test partition persistence."""

    def test_store_and_load_partitions(self, store):
        """Store and load a partition tree."""
        partition_tree = {
            "id": "0",
            "depth": 0,
            "fiedler_value": 0.5,
            "node_ids": ["a", "b", "c", "d"],
            "children": [
                {
                    "id": "0.0",
                    "depth": 1,
                    "fiedler_value": 0.3,
                    "node_ids": ["a", "b"],
                    "children": None,
                },
                {
                    "id": "0.1",
                    "depth": 1,
                    "fiedler_value": 0.4,
                    "node_ids": ["c", "d"],
                    "children": None,
                },
            ],
        }

        store.store_partitions(partition_tree, PROJECT_ID)

        loaded = store.load_partitions(PROJECT_ID)
        assert loaded is not None
        assert loaded["id"] == "0"
        assert loaded["children"] is not None
        assert len(loaded["children"]) == 2
        assert set(loaded["children"][0]["node_ids"]) == {"a", "b"}
        assert set(loaded["children"][1]["node_ids"]) == {"c", "d"}

    def test_no_partitions_returns_none(self, store):
        """Loading from empty store returns None."""
        assert store.load_partitions(PROJECT_ID) is None


class TestFileHashes:
    """Test file hash CRUD."""

    def test_set_and_get_hashes(self, store):
        """Store and retrieve file hashes."""
        hashes = {"mod.py": "abc123", "util.py": "def456"}
        store.set_file_hashes(PROJECT_ID, hashes)

        loaded = store.get_file_hashes(PROJECT_ID)
        assert loaded == hashes

    def test_empty_hashes(self, store):
        """Empty store returns empty dict."""
        assert store.get_file_hashes(PROJECT_ID) == {}

    def test_update_hashes(self, store):
        """Updating hashes replaces old values."""
        store.set_file_hashes(PROJECT_ID, {"a.py": "v1"})
        store.set_file_hashes(PROJECT_ID, {"a.py": "v2", "b.py": "v3"})

        loaded = store.get_file_hashes(PROJECT_ID)
        assert loaded["a.py"] == "v2"
        assert loaded["b.py"] == "v3"


class TestDeleteByFile:
    """Test node deletion by file."""

    def test_delete_nodes_by_file(self, store, sample_graph):
        """Delete all nodes belonging to a specific file."""
        store.store_graph(sample_graph, PROJECT_ID)

        deleted = store.delete_nodes_by_file("util.py", PROJECT_ID)
        assert deleted == 1  # util.helper

        loaded = store.load_graph(PROJECT_ID)
        assert "util.helper" not in loaded.nodes
        assert len(loaded.nodes) == 2  # mod.foo and mod.bar remain

    def test_delete_nonexistent_file(self, store, sample_graph):
        """Deleting nodes from a non-existent file returns 0."""
        store.store_graph(sample_graph, PROJECT_ID)
        assert store.delete_nodes_by_file("missing.py", PROJECT_ID) == 0


class TestStats:
    """Test statistics retrieval."""

    def test_stats_with_data(self, store, sample_graph):
        """Get stats after storing a graph."""
        store.store_graph(sample_graph, PROJECT_ID)

        stats = store.get_stats(PROJECT_ID)
        assert stats["backend"] == "sqlite"
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 3
        assert stats["has_kameda_index"] is False
        assert stats["has_partitions"] is False

    def test_stats_empty(self, store):
        """Get stats from empty store."""
        stats = store.get_stats(PROJECT_ID)
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
