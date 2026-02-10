"""End-to-end tests for M6 graph persistence pipeline.

Tests the full flow: extract → persist → query → modify → incremental update.
"""

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
from curate_ipsum.storage.graph_store import build_graph_store
from curate_ipsum.storage.incremental import IncrementalEngine
from curate_ipsum.storage.synthesis_store import SynthesisStore
from curate_ipsum.synthesis.models import SynthesisResult, SynthesisStatus


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project with Python files."""
    project = tmp_path / "my_project"
    project.mkdir()

    (project / "main.py").write_text("from helper import do_thing\n\ndef run():\n    return do_thing(42)\n")
    (project / "helper.py").write_text("def do_thing(x):\n    return x * 2\n")

    return project


def _make_graph() -> CallGraph:
    """Build a small graph matching the project fixtures."""
    graph = CallGraph()

    graph.add_node(
        GraphNode(
            id="main.run",
            kind=NodeKind.FUNCTION,
            name="run",
            location=SourceLocation(file="main.py", line_start=3, line_end=4),
            signature=FunctionSignature(name="run", params=()),
        )
    )
    graph.add_node(
        GraphNode(
            id="helper.do_thing",
            kind=NodeKind.FUNCTION,
            name="do_thing",
            location=SourceLocation(file="helper.py", line_start=1, line_end=2),
            signature=FunctionSignature(name="do_thing", params=("x",), return_type="int"),
        )
    )

    graph.add_edge(
        GraphEdge(
            source_id="main.run",
            target_id="helper.do_thing",
            kind=EdgeKind.CALLS,
            confidence=1.0,
        )
    )

    return graph


class TestSQLiteEndToEnd:
    """Full M6 pipeline with SQLite backend."""

    def test_extract_persist_query(self, project_dir):
        """Extract → persist → query cycle."""
        store = build_graph_store("sqlite", project_dir)
        project_id = str(project_dir)

        # 1. Persist a graph
        graph = _make_graph()
        store.store_graph(graph, project_id)

        # 2. Query neighbors
        neighbors = store.get_neighbors("main.run", project_id, direction="outgoing")
        assert "helper.do_thing" in neighbors

        # 3. Load full graph back
        loaded = store.load_graph(project_id)
        assert len(loaded.nodes) == 2
        assert len(loaded.edges) == 1

        # 4. Get node details
        node = store.get_node("helper.do_thing", project_id)
        assert node is not None
        assert node["name"] == "do_thing"

        store.close()

    def test_persist_survives_reopen(self, project_dir):
        """Data persists across store close/reopen."""
        project_id = str(project_dir)

        # Store graph
        store1 = build_graph_store("sqlite", project_dir)
        store1.store_graph(_make_graph(), project_id)
        store1.close()

        # Reopen and verify
        store2 = build_graph_store("sqlite", project_dir)
        loaded = store2.load_graph(project_id)
        assert loaded is not None
        assert len(loaded.nodes) == 2
        store2.close()

    def test_incremental_update_after_file_change(self, project_dir):
        """File modification triggers incremental update."""
        project_id = str(project_dir)
        store = build_graph_store("sqlite", project_dir)

        # 1. Initial persist
        graph = _make_graph()
        engine = IncrementalEngine(store)
        result = engine.force_full_rebuild(project_id, graph, project_dir)
        assert result.full_rebuild is True
        assert result.added_nodes == 2

        # 2. Modify a file
        (project_dir / "helper.py").write_text("def do_thing(x):\n    return x * 3  # changed\n")

        # 3. Incremental update (without extractor, just hash tracking + deletion)
        result = engine.update_graph(project_id, project_dir)
        assert result.change_set.modified == ["helper.py"]

        # 4. Verify hashes updated
        hashes = store.get_file_hashes(project_id)
        assert "helper.py" in hashes

        store.close()

    def test_incremental_update_new_file(self, project_dir):
        """Adding a new file is detected."""
        project_id = str(project_dir)
        store = build_graph_store("sqlite", project_dir)

        engine = IncrementalEngine(store)
        # Set initial hashes
        initial_hashes = engine.compute_file_hashes(project_dir)
        store.set_file_hashes(project_id, initial_hashes)

        # Add a new file
        (project_dir / "extra.py").write_text("def extra():\n    pass\n")

        result = engine.update_graph(project_id, project_dir)
        assert "extra.py" in result.change_set.added

        store.close()

    def test_reachability_persistence(self, project_dir):
        """Kameda index survives store close/reopen."""
        project_id = str(project_dir)

        store1 = build_graph_store("sqlite", project_dir)
        kameda = {
            "left_rank": {"main.run": 0, "helper.do_thing": 1},
            "right_rank": {"main.run": 0, "helper.do_thing": 1},
            "source_id": "vs",
            "sink_id": "vt",
            "non_planar_reachability": {},
            "all_node_ids": ["main.run", "helper.do_thing"],
        }
        store1.store_reachability_index(kameda, project_id)
        store1.close()

        # Reopen and query
        store2 = build_graph_store("sqlite", project_dir)
        assert store2.query_reachable("main.run", "helper.do_thing", project_id) is True
        assert store2.query_reachable("helper.do_thing", "main.run", project_id) is False
        store2.close()


class TestSynthesisStorePersistence:
    """Test synthesis results persist across store instances."""

    def test_results_persist(self, tmp_path):
        """Results survive store recreation."""
        store_dir = tmp_path / "synth_data"

        # Write
        store1 = SynthesisStore(store_dir)
        result = SynthesisResult(
            id="persist-test",
            status=SynthesisStatus.SUCCESS,
            iterations=10,
            duration_ms=500,
        )
        store1.append(result, "proj-1")

        # Read with new instance
        store2 = SynthesisStore(store_dir)
        loaded = store2.load_all("proj-1")
        assert len(loaded) == 1
        assert loaded[0].id == "persist-test"
        assert loaded[0].status == SynthesisStatus.SUCCESS

    def test_stats_after_persist(self, project_dir):
        """Stats reflect persisted data."""
        store = build_graph_store("sqlite", project_dir)
        project_id = str(project_dir)

        store.store_graph(_make_graph(), project_id)

        stats = store.get_stats(project_id)
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["has_kameda_index"] is False

        # Store Kameda index
        kameda = {
            "left_rank": {"main.run": 0, "helper.do_thing": 1},
            "right_rank": {"main.run": 0, "helper.do_thing": 1},
            "source_id": "vs",
            "sink_id": "vt",
            "non_planar_reachability": {},
            "all_node_ids": ["main.run", "helper.do_thing"],
        }
        store.store_reachability_index(kameda, project_id)

        stats = store.get_stats(project_id)
        assert stats["has_kameda_index"] is True
        assert stats["kameda_label_count"] == 2

        store.close()
