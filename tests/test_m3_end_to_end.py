"""
End-to-end tests for M3: Belief Revision Engine.

Exit criteria: "Track belief evolution across synthesis attempts with full provenance."

This test exercises the full workflow:
1. Store evidence from mutation testing
2. Add assertions grounded by evidence
3. Query provenance (why_believe, when_added, belief_stability)
4. Analyze a synthesis failure
5. Contract based on failure analysis
6. Rollback to prior state
7. Verify full provenance chain
"""

import pytest

brs = pytest.importorskip("brs")

from brs import Evidence

from theory.assertions import AssertionKind
from theory.failure_analyzer import FailureMode
from theory.manager import TheoryManager
from theory.provenance import RevisionType


@pytest.fixture
def manager(tmp_path):
    """Create a fresh TheoryManager."""
    mgr = TheoryManager(tmp_path)
    mgr._ensure_world_exists()
    return mgr


class TestM3FullWorkflow:
    """End-to-end workflow exercising all M3 features."""

    def test_complete_lifecycle(self, manager):
        """
        Full lifecycle:
        1. Store evidence
        2. Add assertions grounded by evidence
        3. Query provenance
        4. Analyze failure
        5. Contract assertion
        6. Rollback
        """
        # --- Step 1: Store evidence ---
        ev = Evidence(
            id="mut_ev_001",
            citation="mutation test run #1: 85% mutation score",
            kind="mutation_result",
            reliability=0.85,
            date="2025-01-01T00:00:00Z",
            metadata={"score": 0.85, "total": 100, "killed": 85},
        )
        stored_id = manager.store_evidence(ev)
        assert stored_id == "mut_ev_001"

        # --- Step 2: Add assertions ---
        node1 = manager.add_assertion(
            assertion_type="behavior",
            content="sort function returns sorted output",
            evidence_id="mut_ev_001",
            confidence=0.85,
            region_id="file:src/sort.py::func:sort",
        )
        assert node1["id"]
        node1_id = node1["id"]

        node2 = manager.add_assertion(
            assertion_type="postcondition",
            content="result length equals input length",
            evidence_id="mut_ev_001",
            confidence=0.9,
            region_id="file:src/sort.py::func:sort",
        )
        node2_id = node2["id"]

        node3 = manager.add_assertion(
            assertion_type="type",
            content="input is list of comparable elements",
            evidence_id="mut_ev_001",
            confidence=0.7,
            region_id="file:src/sort.py::func:sort",
        )
        node3_id = node3["id"]

        # --- Step 3: Query provenance ---
        # why_believe
        evidence_chain = manager.why_believe(node1_id)
        assert "mut_ev_001" in evidence_chain

        # when_added
        add_event = manager.when_added(node1_id)
        assert add_event is not None
        assert add_event.event_type == RevisionType.EXPAND

        # belief_stability (never revised = 1.0)
        stability = manager.belief_stability(node1_id)
        assert stability == 1.0

        # provenance summary
        summary = manager.get_provenance_summary()
        assert summary["total_events"] >= 4  # 1 evidence + 3 assertions
        assert "expand" in summary["event_type_counts"]

        # --- Step 4: Analyze failure ---
        analysis = manager.analyze_failure(
            error_message="TypeError: '<' not supported between instances of 'str' and 'int'",
            test_pass_rate=0.6,
            mutation_score=0.3,
            region_id="file:src/sort.py::func:sort",
        )
        assert analysis.mode == FailureMode.TYPE_MISMATCH
        assert analysis.confidence > 0
        # Should suggest contracting TYPE assertions
        # (node3 is a TYPE assertion in the same region)

        # --- Step 5: List assertions ---
        assertions = manager.list_assertions(
            region_id="file:src/sort.py::func:sort"
        )
        assert len(assertions) == 3

        # --- Step 6: Create checkpoint before contraction ---
        rb = manager.get_rollback_manager()
        cp = rb.create_checkpoint(
            "before_contraction",
            reason="Save state before contracting type assertion"
        )
        assert cp.world_hash

        # --- Step 7: Contract a type assertion ---
        result = manager.contract_assertion(node3_id, strategy="entrenchment")
        assert node3_id in result.nodes_removed

        # Verify assertion is gone
        remaining = manager.list_assertions(
            region_id="file:src/sort.py::func:sort"
        )
        remaining_ids = [a["id"] for a in remaining]
        assert node3_id not in remaining_ids

        # Check provenance recorded the contraction
        dag_events = manager.provenance_dag.get_history()
        contract_events = [
            e for e in dag_events
            if e.event_type == RevisionType.CONTRACT
        ]
        assert len(contract_events) >= 1
        assert node3_id in contract_events[-1].nodes_removed

        # belief_stability should decrease for contracted assertion
        stability_after = manager.belief_stability(node3_id)
        assert stability_after < 1.0

        # --- Step 8: Rollback to checkpoint ---
        rb.restore_checkpoint("before_contraction")

        # Verify we're back to 3 assertions
        restored = manager.list_assertions(
            region_id="file:src/sort.py::func:sort"
        )
        # Note: after rollback, the world state should include node3 again
        restored_ids = [a["id"] for a in restored]
        assert node3_id in restored_ids
        assert len(restored) == 3

        # --- Step 9: Verify full provenance chain ---
        final_summary = manager.get_provenance_summary()
        assert final_summary["total_events"] >= 6  # evidence + 3 expand + contract + rollback
        assert "rollback" in final_summary["event_type_counts"]


class TestEvidenceToAssertionFlow:
    """Test the evidence → assertion → provenance flow."""

    def test_multiple_evidence_sources(self, manager):
        """Multiple evidence objects grounding different assertions."""
        ev1 = Evidence(
            id="test_ev_1",
            citation="unit tests: 100% pass",
            kind="test_result",
            reliability=0.95,
            date="2025-01-01",
            metadata={},
        )
        ev2 = Evidence(
            id="mut_ev_2",
            citation="mutation tests: 90% killed",
            kind="mutation_result",
            reliability=0.9,
            date="2025-01-02",
            metadata={},
        )

        manager.store_evidence(ev1)
        manager.store_evidence(ev2)

        # Assertions grounded by different evidence
        n1 = manager.add_assertion(
            assertion_type="behavior",
            content="handles edge cases",
            evidence_id="test_ev_1",
            confidence=0.95,
        )
        n2 = manager.add_assertion(
            assertion_type="behavior",
            content="survives mutations",
            evidence_id="mut_ev_2",
            confidence=0.9,
        )

        # Each assertion grounded by its own evidence
        assert "test_ev_1" in manager.why_believe(n1["id"])
        assert "mut_ev_2" in manager.why_believe(n2["id"])

    def test_provenance_world_hashes_grow(self, manager):
        """World hashes should increase as operations are performed."""
        dag = manager.provenance_dag
        initial_hashes = len(dag.get_world_hashes())

        manager.add_assertion(
            assertion_type="type",
            content="x is int",
            evidence_id="ev1",
            confidence=0.5,
        )

        after_hashes = len(dag.get_world_hashes())
        assert after_hashes > initial_hashes


class TestFailureAnalysisIntegration:
    """Test failure analysis with real assertions."""

    def test_analyze_suggests_relevant_contractions(self, manager):
        """Failure analysis should suggest contracting assertions of the right kind."""
        region = "file:src/calc.py::func:divide"

        # Add various assertions
        manager.add_assertion(
            assertion_type="type",
            content="divisor is numeric",
            evidence_id="ev1",
            confidence=0.6,
            region_id=region,
        )
        manager.add_assertion(
            assertion_type="precondition",
            content="divisor is not zero",
            evidence_id="ev2",
            confidence=0.8,
            region_id=region,
        )
        manager.add_assertion(
            assertion_type="behavior",
            content="returns quotient",
            evidence_id="ev3",
            confidence=0.9,
            region_id=region,
        )

        # Analyze a type mismatch failure
        analysis = manager.analyze_failure(
            error_message="TypeError: unsupported operand type(s) for /: 'str' and 'int'",
            region_id=region,
        )

        assert analysis.mode == FailureMode.TYPE_MISMATCH
        # Should suggest contracting the TYPE assertion, not the BEHAVIOR one
        # (contraction suggestions are based on assertion kind matching failure mode)

    def test_analyze_overfitting_detection(self, manager):
        """High test pass + low mutation kill = overfitting."""
        analysis = manager.analyze_failure(
            test_pass_rate=0.98,
            mutation_score=0.15,
        )
        assert analysis.mode == FailureMode.OVERFITTING
        assert analysis.confidence == 0.8


class TestContradictionFlow:
    """Test contradiction detection through the manager."""

    def test_find_type_contradictions(self, manager):
        region = "file:src/main.py::func:process"

        manager.add_assertion(
            assertion_type="type",
            content="x is int",
            evidence_id="ev1",
            confidence=0.7,
            region_id=region,
        )

        contradictions = manager.find_contradictions(
            assertion_type="type",
            content="x is str",
            confidence=0.8,
            region_id=region,
        )

        assert len(contradictions) >= 1

    def test_no_contradictions_different_region(self, manager):
        manager.add_assertion(
            assertion_type="type",
            content="x is int",
            evidence_id="ev1",
            confidence=0.7,
            region_id="file:a.py",
        )

        contradictions = manager.find_contradictions(
            assertion_type="type",
            content="x is str",
            confidence=0.8,
            region_id="file:b.py",
        )

        assert len(contradictions) == 0


class TestRollbackIntegration:
    """Test rollback through the manager."""

    def test_undo_last_operation(self, manager):
        manager.add_assertion(
            assertion_type="behavior",
            content="first assertion",
            evidence_id="ev1",
            confidence=0.5,
        )
        manager.add_assertion(
            assertion_type="behavior",
            content="second assertion",
            evidence_id="ev2",
            confidence=0.6,
        )

        # Count assertions before undo
        before = len(manager.list_assertions())
        assert before == 2

        # Undo last
        rb = manager.get_rollback_manager()
        undone = rb.undo_last(1)
        assert len(undone) == 1

        # Should be back to 1 assertion
        after = len(manager.list_assertions())
        assert after == 1

    def test_world_history(self, manager):
        manager.add_assertion(
            assertion_type="type",
            content="x is int",
            evidence_id="ev1",
            confidence=0.5,
        )

        rb = manager.get_rollback_manager()
        history = rb.list_world_history()
        assert len(history) >= 1

        for world_hash, timestamp, reason in history:
            assert world_hash
            assert timestamp
