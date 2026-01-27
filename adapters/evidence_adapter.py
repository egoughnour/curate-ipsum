"""
Evidence adapter: Maps curate-ipsum run results to py-brs Evidence objects.

This adapter converts TestRunResult and MutationRunResult instances into
BRS Evidence objects that can be used to ground beliefs in the synthesis theory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import MutationRunResult, TestRunResult

try:
    from brs import Evidence
except ImportError:
    # Graceful fallback for when py-brs is not installed
    Evidence = None  # type: ignore[misc, assignment]


class CodeEvidenceKind:
    """Evidence kinds specific to code mutation testing domain."""

    TEST_PASS = "test_pass"
    TEST_FAIL = "test_fail"
    MUTATION_KILLED = "mutation_killed"
    MUTATION_SURVIVED = "mutation_survived"
    MUTATION_NO_COVERAGE = "mutation_no_coverage"


# Reliability mapping based on evidence strength
# A = Strongest (formal proofs, SMT)
# B = Strong (dynamic testing, mutation testing)
# C = Weak (statistical, LLM suggestions)
RELIABILITY_MAP = {
    CodeEvidenceKind.TEST_PASS: "B",
    CodeEvidenceKind.TEST_FAIL: "B",
    CodeEvidenceKind.MUTATION_KILLED: "B",
    CodeEvidenceKind.MUTATION_SURVIVED: "B",
    CodeEvidenceKind.MUTATION_NO_COVERAGE: "C",  # Less reliable - no test coverage
}


def test_result_to_evidence(run: "TestRunResult") -> "Evidence":
    """
    Convert a TestRunResult to a BRS Evidence object.

    Args:
        run: The test run result from curate-ipsum

    Returns:
        A BRS Evidence object that can be used to ground beliefs

    Raises:
        ImportError: If py-brs is not installed
    """
    if Evidence is None:
        raise ImportError("py-brs is required for belief revision. Install with: pip install py-brs>=2.0.0")

    kind = CodeEvidenceKind.TEST_PASS if run.passed else CodeEvidenceKind.TEST_FAIL

    return Evidence(
        id=f"test_{run.id}",
        citation=f"{run.framework} test run at {run.timestamp.isoformat()}",
        kind=kind,
        reliability=RELIABILITY_MAP[kind],
        date=run.timestamp.isoformat(),
        metadata={
            "project_id": run.projectId,
            "commit_sha": run.commitSha,
            "region_id": run.regionId,
            "total_tests": run.totalTests,
            "passed_tests": run.passedTests,
            "failed_tests": run.failedTests,
            "failing_tests": run.failingTests,
            "duration_ms": run.durationMs,
            "framework": run.framework,
            "run_kind": str(run.kind.value),
        },
    )


def mutation_result_to_evidence(run: "MutationRunResult") -> "Evidence":
    """
    Convert a MutationRunResult to a BRS Evidence object.

    The evidence kind is determined by the dominant outcome:
    - MUTATION_KILLED if score > 0.5 (majority killed)
    - MUTATION_SURVIVED if score <= 0.5 (majority survived)
    - MUTATION_NO_COVERAGE if no_coverage > killed + survived

    Args:
        run: The mutation run result from curate-ipsum

    Returns:
        A BRS Evidence object that can be used to ground beliefs

    Raises:
        ImportError: If py-brs is not installed
    """
    if Evidence is None:
        raise ImportError("py-brs is required for belief revision. Install with: pip install py-brs>=2.0.0")

    # Determine primary evidence kind based on outcomes
    if run.noCoverage > (run.killed + run.survived):
        kind = CodeEvidenceKind.MUTATION_NO_COVERAGE
    elif run.mutationScore > 0.5:
        kind = CodeEvidenceKind.MUTATION_KILLED
    else:
        kind = CodeEvidenceKind.MUTATION_SURVIVED

    return Evidence(
        id=f"mutation_{run.id}",
        citation=f"{run.tool} mutation run at {run.timestamp.isoformat()}",
        kind=kind,
        reliability=RELIABILITY_MAP[kind],
        date=run.timestamp.isoformat(),
        metadata={
            "project_id": run.projectId,
            "commit_sha": run.commitSha,
            "region_id": run.regionId,
            "tool": run.tool,
            "total_mutants": run.totalMutants,
            "killed": run.killed,
            "survived": run.survived,
            "no_coverage": run.noCoverage,
            "mutation_score": run.mutationScore,
            "runtime_ms": run.runtimeMs,
            "by_file": [
                {
                    "file_path": f.filePath,
                    "total_mutants": f.totalMutants,
                    "killed": f.killed,
                    "survived": f.survived,
                    "no_coverage": f.noCoverage,
                    "mutation_score": f.mutationScore,
                }
                for f in run.byFile
            ],
        },
    )
