"""Tests for theory.failure_analyzer module."""

import pytest

from curate_ipsum.theory.assertions import Assertion, AssertionKind
from curate_ipsum.theory.failure_analyzer import (
    FailureAnalysis,
    FailureMode,
    FailureModeAnalyzer,
)


class TestFailureMode:
    def test_all_modes_defined(self):
        assert FailureMode.TYPE_MISMATCH == "type_mismatch"
        assert FailureMode.PRECONDITION_VIOLATION == "precondition_violation"
        assert FailureMode.POSTCONDITION_VIOLATION == "postcondition_violation"
        assert FailureMode.INVARIANT_VIOLATION == "invariant_violation"
        assert FailureMode.SEMANTIC_DRIFT == "semantic_drift"
        assert FailureMode.OVERFITTING == "overfitting"
        assert FailureMode.UNDERFITTING == "underfitting"
        assert FailureMode.UNKNOWN == "unknown"


class TestFailureAnalysis:
    def test_to_dict(self):
        fa = FailureAnalysis(
            mode=FailureMode.TYPE_MISMATCH,
            confidence=0.8,
            evidence_summary="TypeError: expected int got str",
            suggested_contraction_ids=["a1", "a2"],
        )
        d = fa.to_dict()
        assert d["failure_mode"] == "type_mismatch"
        assert d["confidence"] == 0.8
        assert d["evidence_summary"] == "TypeError: expected int got str"
        assert d["suggested_contraction_ids"] == ["a1", "a2"]

    def test_to_dict_defaults(self):
        fa = FailureAnalysis(mode=FailureMode.UNKNOWN, confidence=0.3)
        d = fa.to_dict()
        assert d["root_cause_assertion_id"] is None
        assert d["evidence_summary"] == ""
        assert d["suggested_contraction_ids"] == []
        assert d["metadata"] == {}


class TestClassifyError:
    @pytest.mark.parametrize(
        "error_msg,expected_mode",
        [
            ("TypeError: unsupported operand type(s)", FailureMode.TYPE_MISMATCH),
            ("type error in function call", FailureMode.TYPE_MISMATCH),
            ("expected type int but got str", FailureMode.TYPE_MISMATCH),
            ("cannot convert string to float", FailureMode.TYPE_MISMATCH),
            ("incompatible type for argument", FailureMode.TYPE_MISMATCH),
            ("invalid type for parameter", FailureMode.TYPE_MISMATCH),
        ],
    )
    def test_type_error_patterns(self, error_msg, expected_mode):
        assert FailureModeAnalyzer.classify_error(error_msg) == expected_mode

    @pytest.mark.parametrize(
        "error_msg,expected_mode",
        [
            ("precondition not met", FailureMode.PRECONDITION_VIOLATION),
            ("requires x > 0 not met", FailureMode.PRECONDITION_VIOLATION),
            ("invalid argument: must be positive", FailureMode.PRECONDITION_VIOLATION),
            ("ValueError: negative value not allowed", FailureMode.PRECONDITION_VIOLATION),
            ("index out of range", FailureMode.PRECONDITION_VIOLATION),
        ],
    )
    def test_precondition_patterns(self, error_msg, expected_mode):
        assert FailureModeAnalyzer.classify_error(error_msg) == expected_mode

    @pytest.mark.parametrize(
        "error_msg,expected_mode",
        [
            ("postcondition violated", FailureMode.POSTCONDITION_VIOLATION),
            ("ensures result > 0 not met", FailureMode.POSTCONDITION_VIOLATION),
            ("expected 42 but got 0", FailureMode.POSTCONDITION_VIOLATION),
            ("return value incorrect for input", FailureMode.POSTCONDITION_VIOLATION),
        ],
    )
    def test_postcondition_patterns(self, error_msg, expected_mode):
        assert FailureModeAnalyzer.classify_error(error_msg) == expected_mode

    @pytest.mark.parametrize(
        "error_msg,expected_mode",
        [
            ("invariant broken at iteration 5", FailureMode.INVARIANT_VIOLATION),
            ("IndexError: list index out of bounds", FailureMode.INVARIANT_VIOLATION),
            ("infinite loop detected", FailureMode.INVARIANT_VIOLATION),
            ("recursion depth exceeded", FailureMode.INVARIANT_VIOLATION),
            ("stack overflow in recursive call", FailureMode.INVARIANT_VIOLATION),
        ],
    )
    def test_invariant_patterns(self, error_msg, expected_mode):
        assert FailureModeAnalyzer.classify_error(error_msg) == expected_mode

    def test_assertion_error_maps_to_postcondition(self):
        assert (
            FailureModeAnalyzer.classify_error("AssertionError: expected True") == FailureMode.POSTCONDITION_VIOLATION
        )

    def test_empty_message_returns_unknown(self):
        assert FailureModeAnalyzer.classify_error("") == FailureMode.UNKNOWN

    def test_unrecognized_message_returns_unknown(self):
        assert FailureModeAnalyzer.classify_error("something completely unrelated happened") == FailureMode.UNKNOWN

    def test_priority_type_over_precondition(self):
        """Type errors should be detected before precondition violations."""
        msg = "TypeError: invalid argument type"
        assert FailureModeAnalyzer.classify_error(msg) == FailureMode.TYPE_MISMATCH


class TestDetectOverfitting:
    def test_clear_overfitting(self):
        # High test pass, low mutation kill
        assert FailureModeAnalyzer.detect_overfitting(0.95, 0.3) is True

    def test_not_overfitting_balanced(self):
        # Both rates similar
        assert FailureModeAnalyzer.detect_overfitting(0.8, 0.7) is False

    def test_not_overfitting_low_test_rate(self):
        # Low test pass = not overfitting
        assert FailureModeAnalyzer.detect_overfitting(0.3, 0.1) is False

    def test_threshold_boundary(self):
        # Gap clearly below threshold (0.25 < 0.3)
        assert FailureModeAnalyzer.detect_overfitting(0.8, 0.55) is False
        # Gap clearly above threshold (0.4 > 0.3)
        assert FailureModeAnalyzer.detect_overfitting(0.8, 0.4) is True

    def test_custom_threshold(self):
        assert FailureModeAnalyzer.detect_overfitting(0.8, 0.6, overfitting_threshold=0.1) is True


class TestDetectUnderfitting:
    def test_clear_underfitting(self):
        assert FailureModeAnalyzer.detect_underfitting(0.1) is True

    def test_not_underfitting(self):
        assert FailureModeAnalyzer.detect_underfitting(0.8) is False

    def test_boundary(self):
        assert FailureModeAnalyzer.detect_underfitting(0.5) is False
        assert FailureModeAnalyzer.detect_underfitting(0.49) is True

    def test_custom_threshold(self):
        assert FailureModeAnalyzer.detect_underfitting(0.6, underfitting_threshold=0.7) is True


class TestAnalyze:
    def test_analyze_type_error(self):
        result = FailureModeAnalyzer.analyze(
            error_message="TypeError: unsupported operand",
        )
        assert result.mode == FailureMode.TYPE_MISMATCH
        assert result.confidence == 0.7
        assert "TypeError" in result.evidence_summary

    def test_analyze_overfitting(self):
        result = FailureModeAnalyzer.analyze(
            test_pass_rate=0.95,
            mutation_score=0.2,
        )
        assert result.mode == FailureMode.OVERFITTING
        assert result.confidence == 0.8

    def test_analyze_underfitting(self):
        result = FailureModeAnalyzer.analyze(
            test_pass_rate=0.1,
        )
        assert result.mode == FailureMode.UNDERFITTING
        assert result.confidence >= 0.7

    def test_analyze_semantic_drift(self):
        result = FailureModeAnalyzer.analyze(
            test_pass_rate=0.65,
        )
        assert result.mode == FailureMode.SEMANTIC_DRIFT
        assert result.confidence == 0.5

    def test_analyze_unknown(self):
        result = FailureModeAnalyzer.analyze(
            error_message="some random message",
        )
        assert result.mode == FailureMode.UNKNOWN
        assert result.confidence == 0.3

    def test_analyze_with_failing_tests(self):
        result = FailureModeAnalyzer.analyze(
            error_message="AssertionError: values differ",
            failing_tests=["test_add", "test_sub", "test_mul"],
        )
        assert "Failing tests" in result.evidence_summary
        assert result.metadata["failing_test_count"] == 3

    def test_analyze_evidence_summary_truncation(self):
        long_error = "Error: " + "x" * 300
        result = FailureModeAnalyzer.analyze(error_message=long_error)
        assert "..." in result.evidence_summary
        assert len(result.evidence_summary) < 350

    def test_analyze_with_many_failing_tests(self):
        tests = [f"test_{i}" for i in range(10)]
        result = FailureModeAnalyzer.analyze(
            error_message="AssertionError: failed",
            failing_tests=tests,
        )
        assert "+5 more" in result.evidence_summary

    def test_analyze_metadata(self):
        result = FailureModeAnalyzer.analyze(
            test_pass_rate=0.9,
            mutation_score=0.5,
            failing_tests=["test_1"],
        )
        assert result.metadata["test_pass_rate"] == 0.9
        assert result.metadata["mutation_score"] == 0.5
        assert result.metadata["failing_test_count"] == 1

    def test_overfitting_overrides_error_classification(self):
        """Overfitting detection should override error-based classification."""
        result = FailureModeAnalyzer.analyze(
            error_message="TypeError: bad type",
            test_pass_rate=0.95,
            mutation_score=0.2,
        )
        assert result.mode == FailureMode.OVERFITTING


class TestSuggestContractions:
    def _make_assertion(self, id, kind, confidence=0.5, region_id="file:test.py"):
        return Assertion(
            id=id,
            kind=kind,
            content=f"test {id}",
            confidence=confidence,
            region_id=region_id,
        )

    def test_type_mismatch_suggests_type_assertions(self):
        assertions = [
            self._make_assertion("a1", AssertionKind.TYPE, 0.6),
            self._make_assertion("a2", AssertionKind.BEHAVIOR, 0.8),
            self._make_assertion("a3", AssertionKind.TYPE, 0.4),
        ]
        ids = FailureModeAnalyzer._suggest_contractions(FailureMode.TYPE_MISMATCH, assertions, "file:test.py")
        assert "a1" in ids
        assert "a3" in ids
        assert "a2" not in ids
        # Sorted by confidence (weakest first)
        assert ids[0] == "a3"

    def test_postcondition_violation_suggests_postconditions(self):
        assertions = [
            self._make_assertion("a1", AssertionKind.POSTCONDITION, 0.7),
            self._make_assertion("a2", AssertionKind.CONTRACT, 0.5),
            self._make_assertion("a3", AssertionKind.TYPE, 0.3),
        ]
        ids = FailureModeAnalyzer._suggest_contractions(FailureMode.POSTCONDITION_VIOLATION, assertions, "file:test.py")
        assert "a1" in ids
        assert "a2" in ids  # CONTRACT also included
        assert "a3" not in ids

    def test_underfitting_suggests_all(self):
        assertions = [
            self._make_assertion("a1", AssertionKind.TYPE, 0.3),
            self._make_assertion("a2", AssertionKind.BEHAVIOR, 0.7),
            self._make_assertion("a3", AssertionKind.INVARIANT, 0.5),
        ]
        ids = FailureModeAnalyzer._suggest_contractions(FailureMode.UNDERFITTING, assertions, "file:test.py")
        assert len(ids) == 3  # All assertions

    def test_region_filtering(self):
        assertions = [
            self._make_assertion("a1", AssertionKind.TYPE, 0.5, "file:test.py"),
            self._make_assertion("a2", AssertionKind.TYPE, 0.5, "file:other.py"),
        ]
        ids = FailureModeAnalyzer._suggest_contractions(FailureMode.TYPE_MISMATCH, assertions, "file:test.py")
        assert "a1" in ids
        assert "a2" not in ids

    def test_no_region_uses_all(self):
        assertions = [
            self._make_assertion("a1", AssertionKind.TYPE, 0.5, "file:test.py"),
            self._make_assertion("a2", AssertionKind.TYPE, 0.5, "file:other.py"),
        ]
        ids = FailureModeAnalyzer._suggest_contractions(FailureMode.TYPE_MISMATCH, assertions, None)
        assert "a1" in ids
        assert "a2" in ids

    def test_empty_assertions(self):
        ids = FailureModeAnalyzer._suggest_contractions(FailureMode.TYPE_MISMATCH, [], "file:test.py")
        assert ids == []

    def test_unknown_mode_no_suggestions(self):
        assertions = [
            self._make_assertion("a1", AssertionKind.TYPE, 0.5),
        ]
        ids = FailureModeAnalyzer._suggest_contractions(FailureMode.UNKNOWN, assertions, "file:test.py")
        assert ids == []
