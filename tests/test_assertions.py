"""Tests for theory.assertions module."""

import pytest

from theory.assertions import (
    Assertion,
    AssertionKind,
    ContradictionDetector,
    assertion_to_node_dict,
    node_dict_to_assertion,
)


class TestAssertionKind:
    """Test the AssertionKind enum."""

    def test_all_kinds_defined(self):
        assert AssertionKind.TYPE == "type"
        assert AssertionKind.BEHAVIOR == "behavior"
        assert AssertionKind.INVARIANT == "invariant"
        assert AssertionKind.CONTRACT == "contract"
        assert AssertionKind.PRECONDITION == "precondition"
        assert AssertionKind.POSTCONDITION == "postcondition"

    def test_kind_from_string(self):
        assert AssertionKind("type") == AssertionKind.TYPE
        assert AssertionKind("behavior") == AssertionKind.BEHAVIOR

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            AssertionKind("nonexistent")


class TestAssertion:
    """Test the Assertion dataclass."""

    def test_create_valid_assertion(self):
        a = Assertion(
            id="test_1",
            kind=AssertionKind.BEHAVIOR,
            content="function returns positive values",
            confidence=0.8,
            region_id="file:src/main.py",
        )
        assert a.id == "test_1"
        assert a.kind == AssertionKind.BEHAVIOR
        assert a.confidence == 0.8
        assert a.region_id == "file:src/main.py"
        assert a.grounding_evidence_ids == []
        assert a.metadata == {}

    def test_confidence_validation_too_high(self):
        with pytest.raises(ValueError, match="confidence must be between"):
            Assertion(id="x", kind=AssertionKind.TYPE, content="t", confidence=1.5)

    def test_confidence_validation_too_low(self):
        with pytest.raises(ValueError, match="confidence must be between"):
            Assertion(id="x", kind=AssertionKind.TYPE, content="t", confidence=-0.1)

    def test_confidence_boundary_values(self):
        a0 = Assertion(id="x", kind=AssertionKind.TYPE, content="t", confidence=0.0)
        a1 = Assertion(id="y", kind=AssertionKind.TYPE, content="t", confidence=1.0)
        assert a0.confidence == 0.0
        assert a1.confidence == 1.0

    def test_kind_auto_coerce_from_string(self):
        a = Assertion(id="x", kind="behavior", content="t", confidence=0.5)
        assert a.kind == AssertionKind.BEHAVIOR

    def test_created_utc_default(self):
        a = Assertion(id="x", kind=AssertionKind.TYPE, content="t", confidence=0.5)
        assert a.created_utc.endswith("Z")

    def test_metadata_preserved(self):
        a = Assertion(
            id="x",
            kind=AssertionKind.TYPE,
            content="t",
            confidence=0.5,
            metadata={"key": "value"},
        )
        assert a.metadata == {"key": "value"}

    def test_grounding_evidence(self):
        a = Assertion(
            id="x",
            kind=AssertionKind.TYPE,
            content="t",
            confidence=0.5,
            grounding_evidence_ids=["ev1", "ev2"],
        )
        assert a.grounding_evidence_ids == ["ev1", "ev2"]


class TestSerialization:
    """Test assertion_to_node_dict / node_dict_to_assertion round-trip."""

    def test_round_trip(self):
        original = Assertion(
            id="assert_001",
            kind=AssertionKind.BEHAVIOR,
            content="handles null input gracefully",
            confidence=0.75,
            region_id="file:src/utils.py::func:process",
            grounding_evidence_ids=["ev_1", "ev_2"],
            metadata={"source": "mutation_test"},
        )

        node_dict = assertion_to_node_dict(original)
        restored = node_dict_to_assertion(node_dict)

        assert restored.id == original.id
        assert restored.kind == original.kind
        assert restored.content == original.content
        assert restored.confidence == original.confidence
        assert restored.region_id == original.region_id
        assert restored.grounding_evidence_ids == original.grounding_evidence_ids
        assert restored.metadata["source"] == "mutation_test"

    def test_to_node_dict_structure(self):
        a = Assertion(
            id="test_1",
            kind=AssertionKind.TYPE,
            content="x is int",
            confidence=0.9,
        )
        d = assertion_to_node_dict(a)

        assert d["id"] == "test_1"
        assert d["domain_id"] == "code_mutation"
        assert d["kind"] == "Assertion"
        assert d["properties"]["assertion_type"] == "type"
        assert d["properties"]["content"] == "x is int"
        assert d["properties"]["confidence"] == 0.9

    def test_from_node_dict_missing_type_raises(self):
        bad_node = {"id": "x", "properties": {}}
        with pytest.raises(ValueError, match="no assertion_type"):
            node_dict_to_assertion(bad_node)

    def test_from_node_dict_defaults(self):
        node = {
            "id": "test_1",
            "properties": {
                "assertion_type": "invariant",
            },
        }
        a = node_dict_to_assertion(node)
        assert a.id == "test_1"
        assert a.kind == AssertionKind.INVARIANT
        assert a.content == ""
        assert a.confidence == 0.5  # default


class TestContradictionDetector:
    """Test contradiction detection between assertions."""

    def _make(self, id, kind, content, region_id="file:test.py", confidence=0.5):
        return Assertion(
            id=id,
            kind=kind,
            content=content,
            confidence=confidence,
            region_id=region_id,
        )

    def test_type_conflict_same_region(self):
        a = self._make("a", AssertionKind.TYPE, "x: int")
        b = self._make("b", AssertionKind.TYPE, "x: str")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 1
        assert result[0].id == "b"

    def test_type_no_conflict_same_content(self):
        a = self._make("a", AssertionKind.TYPE, "x: int")
        b = self._make("b", AssertionKind.TYPE, "x: int")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 0

    def test_type_no_conflict_different_region(self):
        a = self._make("a", AssertionKind.TYPE, "x: int", region_id="file:a.py")
        b = self._make("b", AssertionKind.TYPE, "x: str", region_id="file:b.py")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 0

    def test_behavior_negation_always_never(self):
        a = self._make("a", AssertionKind.BEHAVIOR, "always returns positive")
        b = self._make("b", AssertionKind.BEHAVIOR, "never returns positive")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 1

    def test_behavior_negation_true_false(self):
        a = self._make("a", AssertionKind.BEHAVIOR, "returns true for valid input")
        b = self._make("b", AssertionKind.BEHAVIOR, "returns false for valid input")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 1

    def test_behavior_no_contradiction_different_content(self):
        a = self._make("a", AssertionKind.BEHAVIOR, "handles null input")
        b = self._make("b", AssertionKind.BEHAVIOR, "logs warnings for errors")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 0

    def test_precondition_contradiction(self):
        a = self._make("a", AssertionKind.PRECONDITION, "input is always valid")
        b = self._make("b", AssertionKind.PRECONDITION, "input is never valid")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 1

    def test_postcondition_contradiction(self):
        a = self._make("a", AssertionKind.POSTCONDITION, "output increases monotonically")
        b = self._make("b", AssertionKind.POSTCONDITION, "output decreases monotonically")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 1

    def test_no_contradiction_same_id(self):
        a = self._make("same", AssertionKind.TYPE, "x: int")
        b = self._make("same", AssertionKind.TYPE, "x: str")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 0  # Same assertion ID = same assertion

    def test_no_contradiction_none_region(self):
        a = Assertion(id="a", kind=AssertionKind.TYPE, content="x: int", confidence=0.5, region_id=None)
        b = Assertion(id="b", kind=AssertionKind.TYPE, content="x: str", confidence=0.5, region_id=None)
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 0  # None region = no conflict

    def test_multiple_contradictions(self):
        new = self._make("new", AssertionKind.BEHAVIOR, "always returns true")
        existing = [
            self._make("e1", AssertionKind.BEHAVIOR, "never returns true"),
            self._make("e2", AssertionKind.BEHAVIOR, "sometimes returns none"),
            self._make("e3", AssertionKind.BEHAVIOR, "never returns true"),
        ]
        result = ContradictionDetector.find_contradictions(new, existing)
        assert len(result) == 2  # e1 and e3 contradict

    def test_cross_kind_no_contradiction(self):
        """TYPE and BEHAVIOR assertions don't contradict each other."""
        a = self._make("a", AssertionKind.TYPE, "always returns int")
        b = self._make("b", AssertionKind.BEHAVIOR, "never returns int")
        result = ContradictionDetector.find_contradictions(a, [b])
        assert len(result) == 0
