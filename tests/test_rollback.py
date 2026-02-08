"""Tests for theory.rollback module."""

import pytest
import tempfile
from pathlib import Path

brs = pytest.importorskip("brs")

from theory.manager import TheoryManager
from theory.provenance import ProvenanceDAG
from theory.rollback import Checkpoint, RollbackError, RollbackManager


@pytest.fixture
def manager(tmp_path):
    """Create a TheoryManager with a temp store."""
    mgr = TheoryManager(tmp_path)
    mgr._ensure_world_exists()
    return mgr


@pytest.fixture
def populated_manager(manager):
    """Manager with some assertions added."""
    manager.add_assertion(
        assertion_type="behavior",
        content="handles null input",
        evidence_id="ev_1",
        confidence=0.8,
    )
    manager.add_assertion(
        assertion_type="type",
        content="x is int",
        evidence_id="ev_2",
        confidence=0.7,
    )
    manager.add_assertion(
        assertion_type="invariant",
        content="loop counter < len",
        evidence_id="ev_3",
        confidence=0.9,
    )
    return manager


class TestCheckpoint:
    def test_create_checkpoint(self):
        cp = Checkpoint(
            name="before_refactor",
            world_hash="abc123",
            timestamp="2025-01-01T00:00:00Z",
            reason="Save state before refactoring",
        )
        assert cp.name == "before_refactor"
        assert cp.world_hash == "abc123"
        assert cp.reason == "Save state before refactoring"

    def test_checkpoint_default_reason(self):
        cp = Checkpoint(name="test", world_hash="h", timestamp="t")
        assert cp.reason == ""


class TestRollbackManager:
    def test_create_checkpoint_and_list(self, populated_manager):
        rb = populated_manager.get_rollback_manager()
        cp = rb.create_checkpoint("before_change", reason="testing")

        assert cp.name == "before_change"
        assert cp.world_hash  # Should have a hash
        assert cp.reason == "testing"

        checkpoints = rb.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].name == "before_change"

    def test_restore_checkpoint(self, populated_manager):
        rb = populated_manager.get_rollback_manager()

        # Create checkpoint
        cp = rb.create_checkpoint("safe_point")
        original_hash = cp.world_hash

        # Add another assertion (changes world)
        populated_manager.add_assertion(
            assertion_type="behavior",
            content="new belief",
            evidence_id="ev_new",
            confidence=0.6,
        )

        # Verify world changed
        current = populated_manager.get_theory_snapshot()
        # The world should have more nodes now

        # Restore checkpoint
        rb.restore_checkpoint("safe_point")

        # Verify we're back at the checkpoint
        row = populated_manager.store._conn.execute(
            "SELECT hash FROM worlds WHERE domain_id=? AND version_label=?",
            (populated_manager.domain, populated_manager.world_label),
        ).fetchone()
        assert row[0] == original_hash

    def test_restore_nonexistent_checkpoint_raises(self, populated_manager):
        rb = populated_manager.get_rollback_manager()
        with pytest.raises(RollbackError, match="not found"):
            rb.restore_checkpoint("does_not_exist")

    def test_rollback_to_invalid_hash_raises(self, populated_manager):
        rb = populated_manager.get_rollback_manager()
        with pytest.raises(RollbackError, match="not found in store"):
            rb.rollback_to("nonexistent_hash_value_12345")

    def test_undo_last(self, populated_manager):
        rb = populated_manager.get_rollback_manager()

        # Get world state before last assertion
        dag = populated_manager.provenance_dag
        events = dag.get_history()
        assert len(events) >= 3  # 3 assertions added

        # Undo the last operation
        undone = rb.undo_last(1)
        assert len(undone) == 1

    def test_undo_multiple(self, populated_manager):
        rb = populated_manager.get_rollback_manager()

        dag = populated_manager.provenance_dag
        events = dag.get_history()
        n_events = len(events)

        # Undo last 2
        undone = rb.undo_last(2)
        assert len(undone) == 2

    def test_undo_too_many_raises(self, populated_manager):
        rb = populated_manager.get_rollback_manager()
        dag = populated_manager.provenance_dag
        n = len(dag.get_history())

        with pytest.raises(RollbackError, match="Cannot undo"):
            rb.undo_last(n + 10)

    def test_list_world_history(self, populated_manager):
        rb = populated_manager.get_rollback_manager()
        history = rb.list_world_history()

        # Should have entries from the 3 assertions we added
        assert len(history) >= 3

        # Each entry is (hash, timestamp, reason)
        for h, ts, reason in history:
            assert h  # non-empty hash
            assert ts  # non-empty timestamp
            assert reason  # non-empty reason

    def test_multiple_checkpoints(self, populated_manager):
        rb = populated_manager.get_rollback_manager()

        cp1 = rb.create_checkpoint("first")
        populated_manager.add_assertion(
            assertion_type="behavior",
            content="mid assertion",
            evidence_id="ev_mid",
            confidence=0.5,
        )
        cp2 = rb.create_checkpoint("second")

        checkpoints = rb.list_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0].name == "first"
        assert checkpoints[1].name == "second"
        assert cp1.world_hash != cp2.world_hash

    def test_rollback_records_provenance_event(self, populated_manager):
        rb = populated_manager.get_rollback_manager()
        cp = rb.create_checkpoint("safe")

        # Add something
        populated_manager.add_assertion(
            assertion_type="type",
            content="y is float",
            evidence_id="ev_y",
            confidence=0.6,
        )

        events_before = len(populated_manager.provenance_dag.get_history())

        # Rollback
        rb.restore_checkpoint("safe")

        events_after = len(populated_manager.provenance_dag.get_history())
        assert events_after > events_before  # Rollback event recorded

        last_event = populated_manager.provenance_dag.get_history()[-1]
        assert last_event.event_type.value == "rollback"
