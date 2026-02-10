"""
Adapters for mapping curate-ipsum types to py-brs types.

This module provides the bridge between curate-ipsum's domain models
(TestRunResult, MutationRunResult, RegionMetrics) and py-brs's
belief revision infrastructure (Evidence, Node, Edge).
"""

from adapters.evidence_adapter import (
    CodeEvidenceKind,
    mutation_result_to_evidence,
    test_result_to_evidence,
)

__all__ = [
    "CodeEvidenceKind",
    "test_result_to_evidence",
    "mutation_result_to_evidence",
]
