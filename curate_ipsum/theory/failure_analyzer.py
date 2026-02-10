"""
Failure mode analyzer for code synthesis.

Classifies why a synthesis attempt failed and maps the failure to
belief revision operations. Uses heuristic pattern matching — formal
verification is M5's concern.

Failure modes (from belief_revision_framework.md):
    - TYPE_MISMATCH: Generated code has type errors
    - PRECONDITION_VIOLATION: Input constraints not met
    - POSTCONDITION_VIOLATION: Output doesn't satisfy specification
    - INVARIANT_VIOLATION: Loop or structural invariant broken
    - SEMANTIC_DRIFT: Code compiles but doesn't do what was intended
    - OVERFITTING: Passes tests but kills few mutants (too specific)
    - UNDERFITTING: Fails basic tests (too general/wrong approach)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from curate_ipsum.theory.assertions import Assertion, AssertionKind


class FailureMode(StrEnum):
    """Classification of why a synthesis attempt failed."""

    TYPE_MISMATCH = "type_mismatch"
    PRECONDITION_VIOLATION = "precondition_violation"
    POSTCONDITION_VIOLATION = "postcondition_violation"
    INVARIANT_VIOLATION = "invariant_violation"
    SEMANTIC_DRIFT = "semantic_drift"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """Result of analyzing a synthesis failure."""

    mode: FailureMode
    confidence: float  # 0.0 to 1.0
    root_cause_assertion_id: str | None = None
    evidence_summary: str = ""
    suggested_contraction_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for MCP tool output."""
        return {
            "failure_mode": self.mode.value,
            "confidence": self.confidence,
            "root_cause_assertion_id": self.root_cause_assertion_id,
            "evidence_summary": self.evidence_summary,
            "suggested_contraction_ids": self.suggested_contraction_ids,
            "metadata": self.metadata,
        }


# Patterns for heuristic classification
_TYPE_ERROR_PATTERNS = [
    re.compile(r"TypeError:", re.IGNORECASE),
    re.compile(r"type\s+error", re.IGNORECASE),
    re.compile(r"expected\s+type\s+\w+.*got\s+\w+", re.IGNORECASE),
    re.compile(r"cannot\s+convert", re.IGNORECASE),
    re.compile(r"incompatible\s+type", re.IGNORECASE),
    re.compile(r"invalid\s+type", re.IGNORECASE),
]

_ASSERTION_ERROR_PATTERNS = [
    re.compile(r"AssertionError:", re.IGNORECASE),
    re.compile(r"assert\s+.*failed", re.IGNORECASE),
]

_PRECONDITION_PATTERNS = [
    re.compile(r"precondition", re.IGNORECASE),
    re.compile(r"requires?\s+.*not\s+met", re.IGNORECASE),
    re.compile(r"invalid\s+(argument|input|parameter)", re.IGNORECASE),
    re.compile(r"ValueError:", re.IGNORECASE),
    re.compile(r"out\s+of\s+range", re.IGNORECASE),
]

_POSTCONDITION_PATTERNS = [
    re.compile(r"postcondition", re.IGNORECASE),
    re.compile(r"ensures?\s+.*not\s+met", re.IGNORECASE),
    re.compile(r"expected\s+.*but\s+got", re.IGNORECASE),
    re.compile(r"return.*value.*incorrect", re.IGNORECASE),
]

_INVARIANT_PATTERNS = [
    re.compile(r"invariant", re.IGNORECASE),
    re.compile(r"IndexError:", re.IGNORECASE),
    re.compile(r"index\s+out\s+of\s+(bounds|range)", re.IGNORECASE),
    re.compile(r"infinite\s+loop", re.IGNORECASE),
    re.compile(r"recursion.*depth.*exceeded", re.IGNORECASE),
    re.compile(r"stack\s+overflow", re.IGNORECASE),
]


class FailureModeAnalyzer:
    """
    Heuristic analyzer for synthesis failure classification.

    Examines error messages, test results, and mutation results to
    determine why a generated patch failed and suggests which beliefs
    should be contracted to avoid future failures.
    """

    @staticmethod
    def classify_error(error_message: str) -> FailureMode:
        """
        Classify a failure based on error message patterns.

        Args:
            error_message: The error output from test execution

        Returns:
            The most likely FailureMode
        """
        if not error_message:
            return FailureMode.UNKNOWN

        # Check patterns in priority order
        for pattern in _TYPE_ERROR_PATTERNS:
            if pattern.search(error_message):
                return FailureMode.TYPE_MISMATCH

        for pattern in _PRECONDITION_PATTERNS:
            if pattern.search(error_message):
                return FailureMode.PRECONDITION_VIOLATION

        for pattern in _POSTCONDITION_PATTERNS:
            if pattern.search(error_message):
                return FailureMode.POSTCONDITION_VIOLATION

        for pattern in _INVARIANT_PATTERNS:
            if pattern.search(error_message):
                return FailureMode.INVARIANT_VIOLATION

        for pattern in _ASSERTION_ERROR_PATTERNS:
            if pattern.search(error_message):
                return FailureMode.POSTCONDITION_VIOLATION

        return FailureMode.UNKNOWN

    @staticmethod
    def detect_overfitting(
        test_pass_rate: float,
        mutation_score: float,
        overfitting_threshold: float = 0.3,
    ) -> bool:
        """
        Detect if a patch is overfitting to tests.

        Overfitting = high test pass rate but low mutation kill rate.
        The code passes existing tests but doesn't actually implement
        the correct behavior (just gets lucky on the test cases).

        Args:
            test_pass_rate: Fraction of tests passing (0.0 to 1.0)
            mutation_score: Fraction of mutants killed (0.0 to 1.0)
            overfitting_threshold: Gap threshold for overfitting detection

        Returns:
            True if overfitting detected
        """
        if test_pass_rate < 0.5:
            return False  # Not passing enough tests to be overfitting

        gap = test_pass_rate - mutation_score
        return gap > overfitting_threshold

    @staticmethod
    def detect_underfitting(
        test_pass_rate: float,
        underfitting_threshold: float = 0.5,
    ) -> bool:
        """
        Detect if a patch is underfitting.

        Underfitting = failing basic tests. The generated code doesn't
        even satisfy fundamental requirements.

        Args:
            test_pass_rate: Fraction of tests passing (0.0 to 1.0)
            underfitting_threshold: Pass rate below which underfitting is declared

        Returns:
            True if underfitting detected
        """
        return test_pass_rate < underfitting_threshold

    @classmethod
    def analyze(
        cls,
        error_message: str = "",
        test_pass_rate: float | None = None,
        mutation_score: float | None = None,
        failing_tests: list[str] | None = None,
        assertions: list[Assertion] | None = None,
        region_id: str | None = None,
    ) -> FailureAnalysis:
        """
        Full failure analysis combining all heuristics.

        Args:
            error_message: Error output from test execution
            test_pass_rate: Fraction of tests passing (0.0 to 1.0)
            mutation_score: Fraction of mutants killed (0.0 to 1.0)
            failing_tests: Names of failing tests
            assertions: Current assertions in the theory
            region_id: Region where the patch was applied

        Returns:
            FailureAnalysis with mode, confidence, and suggestions
        """
        # Step 1: Classify error message
        mode = cls.classify_error(error_message)
        confidence = 0.7 if mode != FailureMode.UNKNOWN else 0.3

        # Step 2: Check for overfitting/underfitting
        if test_pass_rate is not None and mutation_score is not None:
            if cls.detect_overfitting(test_pass_rate, mutation_score):
                mode = FailureMode.OVERFITTING
                confidence = 0.8
            elif cls.detect_underfitting(test_pass_rate):
                mode = FailureMode.UNDERFITTING
                confidence = 0.8
        elif test_pass_rate is not None:
            if cls.detect_underfitting(test_pass_rate):
                mode = FailureMode.UNDERFITTING
                confidence = 0.7

        # Step 3: If still unknown, try semantic drift detection
        if mode == FailureMode.UNKNOWN and test_pass_rate is not None:
            if 0.5 <= test_pass_rate < 0.8:
                mode = FailureMode.SEMANTIC_DRIFT
                confidence = 0.5

        # Step 4: Build evidence summary
        evidence_parts = []
        if error_message:
            # Truncate for readability
            short_error = error_message[:200] + ("..." if len(error_message) > 200 else "")
            evidence_parts.append(f"Error: {short_error}")
        if test_pass_rate is not None:
            evidence_parts.append(f"Test pass rate: {test_pass_rate:.1%}")
        if mutation_score is not None:
            evidence_parts.append(f"Mutation score: {mutation_score:.1%}")
        if failing_tests:
            evidence_parts.append(
                f"Failing tests: {', '.join(failing_tests[:5])}"
                + (f" (+{len(failing_tests) - 5} more)" if len(failing_tests) > 5 else "")
            )

        # Step 5: Suggest contraction targets
        suggested_ids = cls._suggest_contractions(mode, assertions or [], region_id)

        return FailureAnalysis(
            mode=mode,
            confidence=confidence,
            evidence_summary="; ".join(evidence_parts),
            suggested_contraction_ids=suggested_ids,
            metadata={
                "test_pass_rate": test_pass_rate,
                "mutation_score": mutation_score,
                "failing_test_count": len(failing_tests) if failing_tests else 0,
            },
        )

    @staticmethod
    def _suggest_contractions(
        mode: FailureMode,
        assertions: list[Assertion],
        region_id: str | None,
    ) -> list[str]:
        """
        Suggest which assertions should be contracted based on failure mode.

        The mapping from failure mode to assertion kind:
        - TYPE_MISMATCH → contract TYPE assertions in the region
        - PRECONDITION_VIOLATION → contract PRECONDITION assertions
        - POSTCONDITION_VIOLATION → contract POSTCONDITION assertions
        - INVARIANT_VIOLATION → contract INVARIANT assertions
        - SEMANTIC_DRIFT → contract BEHAVIOR assertions (weakest first)
        - OVERFITTING → contract BEHAVIOR assertions with lowest confidence
        - UNDERFITTING → contract all assertions in region (start fresh)
        """
        # Filter to assertions in the relevant region
        if region_id:
            region_assertions = [a for a in assertions if a.region_id == region_id]
        else:
            region_assertions = assertions

        if not region_assertions:
            return []

        # Map failure mode to target assertion kinds
        target_kinds: dict[FailureMode, list[AssertionKind]] = {
            FailureMode.TYPE_MISMATCH: [AssertionKind.TYPE],
            FailureMode.PRECONDITION_VIOLATION: [
                AssertionKind.PRECONDITION,
                AssertionKind.CONTRACT,
            ],
            FailureMode.POSTCONDITION_VIOLATION: [
                AssertionKind.POSTCONDITION,
                AssertionKind.CONTRACT,
            ],
            FailureMode.INVARIANT_VIOLATION: [AssertionKind.INVARIANT],
            FailureMode.SEMANTIC_DRIFT: [AssertionKind.BEHAVIOR],
            FailureMode.OVERFITTING: [AssertionKind.BEHAVIOR],
            FailureMode.UNDERFITTING: list(AssertionKind),  # All kinds
        }

        kinds = target_kinds.get(mode, [])
        candidates = [a for a in region_assertions if a.kind in kinds]

        # Sort by confidence (weakest first — contract least-entrenched beliefs)
        candidates.sort(key=lambda a: a.confidence)

        return [a.id for a in candidates]
