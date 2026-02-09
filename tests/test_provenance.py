"""Tests for theory.provenance module."""

from theory.provenance import (
    ProvenanceDAG,
    RevisionEvent,
    RevisionType,
)


def _make_event(
    event_type=RevisionType.EXPAND,
    assertion_id="a1",
    evidence_id="ev1",
    from_hash="hash_0",
    to_hash="hash_1",
    reason="test",
    nodes_removed=None,
    nodes_added=None,
    strategy=None,
):
    return RevisionEvent(
        event_type=event_type,
        timestamp="2025-01-01T00:00:00Z",
        assertion_id=assertion_id,
        evidence_id=evidence_id,
        from_world_hash=from_hash,
        to_world_hash=to_hash,
        reason=reason,
        nodes_removed=nodes_removed or [],
        nodes_added=nodes_added or [],
        strategy=strategy,
    )


class TestRevisionType:
    def test_all_types(self):
        assert RevisionType.EXPAND == "expand"
        assert RevisionType.CONTRACT == "contract"
        assert RevisionType.REVISE == "revise"
        assert RevisionType.EVIDENCE == "evidence"
        assert RevisionType.ROLLBACK == "rollback"


class TestRevisionEvent:
    def test_create_event(self):
        e = _make_event()
        assert e.event_type == RevisionType.EXPAND
        assert e.assertion_id == "a1"
        assert e.evidence_id == "ev1"
        assert e.from_world_hash == "hash_0"
        assert e.to_world_hash == "hash_1"

    def test_to_dict(self):
        e = _make_event(strategy="entrenchment")
        d = e.to_dict()
        assert d["event_type"] == "expand"
        assert d["assertion_id"] == "a1"
        assert d["strategy"] == "entrenchment"

    def test_from_dict_round_trip(self):
        original = _make_event(
            nodes_added=["n1", "n2"],
        )
        original.metadata = {"key": "val"}
        d = original.to_dict()
        restored = RevisionEvent.from_dict(d)
        assert restored.event_type == original.event_type
        assert restored.assertion_id == original.assertion_id
        assert restored.nodes_added == ["n1", "n2"]
        assert restored.metadata == {"key": "val"}


class TestProvenanceDAG:
    def test_empty_dag(self):
        dag = ProvenanceDAG()
        assert dag.get_history() == []
        assert dag.events == []

    def test_add_single_event(self):
        dag = ProvenanceDAG()
        e = _make_event()
        dag.add_event(e)
        assert len(dag.get_history()) == 1
        assert dag.get_history()[0].assertion_id == "a1"

    def test_add_multiple_events(self):
        dag = ProvenanceDAG()
        dag.add_event(_make_event(assertion_id="a1", to_hash="h1"))
        dag.add_event(_make_event(assertion_id="a2", from_hash="h1", to_hash="h2"))
        dag.add_event(_make_event(assertion_id="a3", from_hash="h2", to_hash="h3"))
        assert len(dag.get_history()) == 3

    def test_get_path_connected(self):
        dag = ProvenanceDAG()
        dag.add_event(_make_event(from_hash="h0", to_hash="h1"))
        dag.add_event(_make_event(from_hash="h1", to_hash="h2"))
        dag.add_event(_make_event(from_hash="h2", to_hash="h3"))

        path = dag.get_path("h0", "h3")
        assert len(path) == 3
        assert path[0].from_world_hash == "h0"
        assert path[-1].to_world_hash == "h3"

    def test_get_path_no_connection(self):
        dag = ProvenanceDAG()
        dag.add_event(_make_event(from_hash="h0", to_hash="h1"))
        dag.add_event(_make_event(from_hash="h5", to_hash="h6"))

        path = dag.get_path("h0", "h6")
        assert path == []

    def test_get_path_same_hash(self):
        dag = ProvenanceDAG()
        dag.add_event(_make_event(from_hash="h0", to_hash="h1"))
        path = dag.get_path("h0", "h0")
        assert path == []

    def test_why_believe(self):
        dag = ProvenanceDAG()
        dag.add_event(
            _make_event(
                event_type=RevisionType.EXPAND,
                assertion_id="a1",
                evidence_id="ev1",
            )
        )
        dag.add_event(
            _make_event(
                event_type=RevisionType.REVISE,
                assertion_id="a1",
                evidence_id="ev2",
                from_hash="h1",
                to_hash="h2",
            )
        )
        # Evidence event shouldn't be counted
        dag.add_event(
            _make_event(
                event_type=RevisionType.EVIDENCE,
                assertion_id="a1",
                evidence_id="ev3",
                from_hash="h2",
                to_hash="h3",
            )
        )

        evidence = dag.why_believe("a1")
        assert "ev1" in evidence
        assert "ev2" in evidence
        assert "ev3" not in evidence  # EVIDENCE events don't ground assertions

    def test_why_believe_unknown_assertion(self):
        dag = ProvenanceDAG()
        assert dag.why_believe("nonexistent") == []

    def test_when_added(self):
        dag = ProvenanceDAG()
        dag.add_event(
            _make_event(
                event_type=RevisionType.EXPAND,
                assertion_id="a1",
                to_hash="h1",
            )
        )
        dag.add_event(
            _make_event(
                event_type=RevisionType.CONTRACT,
                assertion_id="a2",
                to_hash="h2",
                nodes_removed=["a1"],
            )
        )

        event = dag.when_added("a1")
        assert event is not None
        assert event.event_type == RevisionType.EXPAND

    def test_when_added_via_nodes_added(self):
        dag = ProvenanceDAG()
        dag.add_event(
            _make_event(
                event_type=RevisionType.REVISE,
                assertion_id="a_main",
                nodes_added=["a_side"],
                to_hash="h1",
            )
        )

        event = dag.when_added("a_side")
        assert event is not None
        assert event.event_type == RevisionType.REVISE

    def test_when_added_not_found(self):
        dag = ProvenanceDAG()
        assert dag.when_added("nonexistent") is None

    def test_when_removed(self):
        dag = ProvenanceDAG()
        dag.add_event(
            _make_event(
                event_type=RevisionType.EXPAND,
                assertion_id="a1",
                to_hash="h1",
            )
        )
        dag.add_event(
            _make_event(
                event_type=RevisionType.CONTRACT,
                assertion_id="a1",
                to_hash="h2",
                nodes_removed=["a1"],
            )
        )

        event = dag.when_removed("a1")
        assert event is not None
        assert event.event_type == RevisionType.CONTRACT

    def test_when_removed_still_present(self):
        dag = ProvenanceDAG()
        dag.add_event(
            _make_event(
                event_type=RevisionType.EXPAND,
                assertion_id="a1",
            )
        )
        assert dag.when_removed("a1") is None

    def test_belief_stability_never_revised(self):
        dag = ProvenanceDAG()
        dag.add_event(
            _make_event(
                event_type=RevisionType.EXPAND,
                assertion_id="a1",
            )
        )
        assert dag.belief_stability("a1") == 1.0

    def test_belief_stability_once_removed(self):
        dag = ProvenanceDAG()
        dag.add_event(
            _make_event(
                event_type=RevisionType.EXPAND,
                assertion_id="a1",
                to_hash="h1",
            )
        )
        dag.add_event(
            _make_event(
                event_type=RevisionType.CONTRACT,
                assertion_id="a1",
                to_hash="h2",
                nodes_removed=["a1"],
            )
        )
        # 1 / (1 + 1) = 0.5
        assert dag.belief_stability("a1") == 0.5

    def test_belief_stability_unknown(self):
        dag = ProvenanceDAG()
        # Unknown assertion assumed stable
        assert dag.belief_stability("nonexistent") == 1.0

    def test_get_world_hashes(self):
        dag = ProvenanceDAG()
        dag.add_event(_make_event(from_hash="h0", to_hash="h1"))
        dag.add_event(_make_event(from_hash="h1", to_hash="h2"))
        dag.add_event(_make_event(from_hash="h2", to_hash="h3"))

        hashes = dag.get_world_hashes()
        assert hashes == ["h0", "h1", "h2", "h3"]

    def test_get_world_hashes_dedup(self):
        dag = ProvenanceDAG()
        dag.add_event(_make_event(from_hash="h0", to_hash="h1"))
        dag.add_event(_make_event(from_hash="h0", to_hash="h2"))

        hashes = dag.get_world_hashes()
        assert hashes.count("h0") == 1  # No duplicates

    def test_to_dict_from_dict_round_trip(self):
        dag = ProvenanceDAG()
        dag.add_event(_make_event(assertion_id="a1", from_hash="h0", to_hash="h1"))
        dag.add_event(
            _make_event(
                event_type=RevisionType.CONTRACT,
                assertion_id="a2",
                from_hash="h1",
                to_hash="h2",
                nodes_removed=["a1"],
            )
        )

        data = dag.to_dict()
        restored = ProvenanceDAG.from_dict(data)

        assert len(restored.get_history()) == 2
        assert restored.get_history()[0].assertion_id == "a1"
        assert restored.get_history()[1].nodes_removed == ["a1"]

    def test_events_property_returns_copy(self):
        dag = ProvenanceDAG()
        dag.add_event(_make_event())
        events = dag.events
        events.clear()
        assert len(dag.events) == 1  # Original not affected

    def test_index_by_assertion(self):
        """Verify internal indexing works for assertion lookups."""
        dag = ProvenanceDAG()
        dag.add_event(
            _make_event(
                event_type=RevisionType.EXPAND,
                assertion_id="a1",
                evidence_id="ev1",
            )
        )
        dag.add_event(
            _make_event(
                event_type=RevisionType.EXPAND,
                assertion_id="a2",
                evidence_id="ev2",
            )
        )

        # why_believe uses the index
        assert dag.why_believe("a1") == ["ev1"]
        assert dag.why_believe("a2") == ["ev2"]

    def test_index_nodes_added(self):
        """Nodes in nodes_added should also be indexed."""
        dag = ProvenanceDAG()
        dag.add_event(
            _make_event(
                event_type=RevisionType.REVISE,
                assertion_id="main",
                nodes_added=["side_a", "side_b"],
                evidence_id="ev1",
            )
        )

        # side_a should be traceable
        assert dag.when_added("side_a") is not None
