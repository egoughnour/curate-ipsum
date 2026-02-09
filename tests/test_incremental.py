"""Tests for storage.incremental — Incremental update engine."""

import hashlib

import pytest

from graph.models import (
    CallGraph,
    GraphNode,
    NodeKind,
    SourceLocation,
)
from storage.incremental import ChangeSet, IncrementalEngine
from storage.sqlite_graph_store import SQLiteGraphStore


@pytest.fixture
def store(tmp_path):
    """Create a SQLiteGraphStore for testing."""
    db_path = tmp_path / "test_graph.db"
    s = SQLiteGraphStore(db_path)
    yield s
    s.close()


@pytest.fixture
def project_dir(tmp_path):
    """Create a temp directory with some Python files."""
    project = tmp_path / "project"
    project.mkdir()

    (project / "mod.py").write_text("def foo():\n    return 1\n")
    (project / "util.py").write_text("def helper():\n    return 2\n")
    (project / "lib.py").write_text("def lib_func():\n    return 3\n")

    return project


PROJECT_ID = "inc-test"


class TestComputeFileHashes:
    """Test file hash computation."""

    def test_basic_hashing(self, project_dir):
        """Compute hashes for Python files."""
        hashes = IncrementalEngine.compute_file_hashes(project_dir)

        assert "mod.py" in hashes
        assert "util.py" in hashes
        assert "lib.py" in hashes
        assert len(hashes) == 3

        # Verify hash is correct SHA-256
        content = (project_dir / "mod.py").read_bytes()
        expected = hashlib.sha256(content).hexdigest()
        assert hashes["mod.py"] == expected

    def test_pattern_filter(self, project_dir):
        """Pattern filters files correctly."""
        # Add a non-.py file
        (project_dir / "readme.txt").write_text("hello")

        hashes = IncrementalEngine.compute_file_hashes(project_dir, "**/*.py")
        assert "readme.txt" not in hashes
        assert "mod.py" in hashes

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty dict."""
        empty = tmp_path / "empty"
        empty.mkdir()
        assert IncrementalEngine.compute_file_hashes(empty) == {}


class TestDetectChanges:
    """Test change detection."""

    def test_detect_added_files(self, store, project_dir):
        """Detect newly added files."""
        engine = IncrementalEngine(store)

        # First scan — no stored hashes, so all files are "added"
        hashes = engine.compute_file_hashes(project_dir)
        changes = engine.detect_changes(PROJECT_ID, hashes)

        assert len(changes.added) == 3
        assert len(changes.modified) == 0
        assert len(changes.removed) == 0

    def test_detect_modified_files(self, store, project_dir):
        """Detect modified files."""
        engine = IncrementalEngine(store)

        # Store initial hashes
        hashes_v1 = engine.compute_file_hashes(project_dir)
        store.set_file_hashes(PROJECT_ID, hashes_v1)

        # Modify a file
        (project_dir / "mod.py").write_text("def foo():\n    return 99\n")

        hashes_v2 = engine.compute_file_hashes(project_dir)
        changes = engine.detect_changes(PROJECT_ID, hashes_v2)

        assert changes.modified == ["mod.py"]
        assert changes.added == []
        assert changes.removed == []

    def test_detect_removed_files(self, store, project_dir):
        """Detect removed files."""
        engine = IncrementalEngine(store)

        hashes_v1 = engine.compute_file_hashes(project_dir)
        store.set_file_hashes(PROJECT_ID, hashes_v1)

        # Remove a file
        (project_dir / "lib.py").unlink()

        hashes_v2 = engine.compute_file_hashes(project_dir)
        changes = engine.detect_changes(PROJECT_ID, hashes_v2)

        assert changes.removed == ["lib.py"]
        assert changes.added == []
        assert changes.modified == []

    def test_no_changes(self, store, project_dir):
        """No changes detected when files haven't changed."""
        engine = IncrementalEngine(store)

        hashes = engine.compute_file_hashes(project_dir)
        store.set_file_hashes(PROJECT_ID, hashes)

        # Same hashes
        changes = engine.detect_changes(PROJECT_ID, hashes)
        assert not changes.has_changes

    def test_combined_changes(self, store, project_dir):
        """Detect a mix of added, modified, and removed files."""
        engine = IncrementalEngine(store)

        hashes_v1 = engine.compute_file_hashes(project_dir)
        store.set_file_hashes(PROJECT_ID, hashes_v1)

        # Modify mod.py, remove lib.py, add new.py
        (project_dir / "mod.py").write_text("def foo():\n    return 99\n")
        (project_dir / "lib.py").unlink()
        (project_dir / "new.py").write_text("def new_func():\n    pass\n")

        hashes_v2 = engine.compute_file_hashes(project_dir)
        changes = engine.detect_changes(PROJECT_ID, hashes_v2)

        assert changes.added == ["new.py"]
        assert changes.modified == ["mod.py"]
        assert changes.removed == ["lib.py"]
        assert changes.total_changed == 3


class TestUpdateGraph:
    """Test incremental graph update."""

    def test_update_removes_deleted_file_nodes(self, store, project_dir):
        """Incremental update removes nodes for deleted files."""
        # Pre-populate graph with nodes for each file
        graph = CallGraph()
        graph.add_node(
            GraphNode(
                id="mod.foo",
                kind=NodeKind.FUNCTION,
                name="foo",
                location=SourceLocation(file="mod.py", line_start=1, line_end=2),
            )
        )
        graph.add_node(
            GraphNode(
                id="lib.func",
                kind=NodeKind.FUNCTION,
                name="lib_func",
                location=SourceLocation(file="lib.py", line_start=1, line_end=2),
            )
        )
        store.store_graph(graph, PROJECT_ID)

        # Set initial hashes
        hashes = IncrementalEngine.compute_file_hashes(project_dir)
        store.set_file_hashes(PROJECT_ID, hashes)

        # Delete lib.py
        (project_dir / "lib.py").unlink()

        engine = IncrementalEngine(store)
        result = engine.update_graph(PROJECT_ID, project_dir)

        assert result.change_set.removed == ["lib.py"]
        assert result.removed_nodes == 1

        # Verify lib.func is gone from store
        assert store.get_node("lib.func", PROJECT_ID) is None
        # mod.foo should remain
        assert store.get_node("mod.foo", PROJECT_ID) is not None

    def test_update_no_changes(self, store, project_dir):
        """Update with no changes is fast and returns zero counts."""
        engine = IncrementalEngine(store)

        hashes = engine.compute_file_hashes(project_dir)
        store.set_file_hashes(PROJECT_ID, hashes)

        result = engine.update_graph(PROJECT_ID, project_dir)
        assert result.added_nodes == 0
        assert result.removed_nodes == 0
        assert result.modified_files == 0


class TestForceFullRebuild:
    """Test forced full rebuild."""

    def test_full_rebuild(self, store, project_dir):
        """Force rebuild stores entire graph and hashes."""
        graph = CallGraph()
        graph.add_node(
            GraphNode(
                id="a",
                kind=NodeKind.FUNCTION,
                name="a",
                location=SourceLocation(file="mod.py", line_start=1, line_end=1),
            )
        )
        graph.add_node(
            GraphNode(
                id="b",
                kind=NodeKind.FUNCTION,
                name="b",
                location=SourceLocation(file="util.py", line_start=1, line_end=1),
            )
        )

        engine = IncrementalEngine(store)
        result = engine.force_full_rebuild(PROJECT_ID, graph, project_dir)

        assert result.full_rebuild is True
        assert result.added_nodes == 2

        # Verify graph persisted
        loaded = store.load_graph(PROJECT_ID)
        assert loaded is not None
        assert len(loaded.nodes) == 2

        # Verify hashes stored
        hashes = store.get_file_hashes(PROJECT_ID)
        assert "mod.py" in hashes


class TestChangeSet:
    """Test ChangeSet data class."""

    def test_has_changes(self):
        assert ChangeSet().has_changes is False
        assert ChangeSet(added=["a"]).has_changes is True
        assert ChangeSet(removed=["b"]).has_changes is True

    def test_total_changed(self):
        cs = ChangeSet(added=["a", "b"], modified=["c"], removed=["d"])
        assert cs.total_changed == 4

    def test_to_dict(self):
        cs = ChangeSet(added=["a"], modified=["b"], removed=["c"])
        d = cs.to_dict()
        assert d["added"] == ["a"]
        assert d["total_changed"] == 3
