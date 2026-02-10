"""
Stryker mutation testing report parser.

Stryker is a mutation testing framework primarily for JavaScript/TypeScript.
This parser handles Stryker's JSON report format.

Report locations searched (in order):
1. Explicit reportPath argument
2. reports/mutation/mutation.json
3. reports/stryker-report.json
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

from curate_ipsum.models import FileMutationStats

LOG = logging.getLogger("parsers.stryker")

DEFAULT_STRYKER_REPORT = os.environ.get("MUTATION_TOOL_STRYKER_REPORT", "reports/mutation/mutation.json")


def _compute_mutation_score(killed: int, survived: int, no_coverage: int) -> float:
    """
    Compute mutation score as killed / (killed + survived).

    No-coverage mutants are excluded from the denominator as they
    represent gaps in test coverage rather than mutation detection.
    """
    denominator = killed + survived
    if denominator == 0:
        return 0.0
    return killed / float(denominator)


def _collect_mutants(data: dict) -> dict[str, list[dict]]:
    """
    Extract mutants from Stryker report, handling different report formats.

    Stryker has evolved its report format over versions. This handles:
    - files.{path}.mutants format (newer)
    - mutantResults array format (older)
    """
    files: dict[str, list[dict]] = {}

    if "files" in data and isinstance(data["files"], dict):
        # Newer format: files grouped by path
        for path, file_info in data["files"].items():
            mutants = file_info.get("mutants", [])
            if isinstance(mutants, list):
                files[path] = mutants
    elif "mutantResults" in data and isinstance(data["mutantResults"], list):
        # Older format: flat list of mutants
        files["<all>"] = data["mutantResults"]

    return files


def _count_mutants(mutants: Iterable[dict]) -> tuple[int, int, int]:
    """
    Count mutants by status.

    Returns: (killed, survived, no_coverage)
    """
    killed = survived = no_coverage = 0

    for mutant in mutants:
        status = str(mutant.get("status", "")).lower()
        if status == "killed":
            killed += 1
        elif status == "survived":
            survived += 1
        elif status in ("no coverage", "nocoverage", "survivednocoverage"):
            no_coverage += 1
        # Other statuses (timeout, error, etc.) are not counted

    return killed, survived, no_coverage


def find_stryker_report(
    working_directory: str,
    report_path: str | None = None,
) -> Path | None:
    """
    Locate the Stryker report file.

    Args:
        working_directory: Project directory
        report_path: Optional explicit path to report

    Returns:
        Path to report file, or None if not found
    """
    cwd_path = Path(working_directory)
    candidate_paths: list[Path] = []

    # Explicit path takes priority
    if report_path:
        explicit_path = Path(report_path)
        candidate_paths.append(explicit_path)
        if not explicit_path.is_absolute():
            candidate_paths.append(cwd_path / explicit_path)

    # Default locations
    if DEFAULT_STRYKER_REPORT:
        candidate_paths.append(cwd_path / DEFAULT_STRYKER_REPORT)

    candidate_paths.extend(
        [
            cwd_path / "reports" / "mutation" / "mutation.json",
            cwd_path / "reports" / "stryker-report.json",
            cwd_path / "stryker-report.json",
        ]
    )

    return next((p for p in candidate_paths if p.exists()), None)


def parse_stryker_report(report_path: Path) -> dict:
    """
    Parse a Stryker JSON report file.

    Args:
        report_path: Path to the report file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If report doesn't exist
        ValueError: If report is not valid JSON or wrong format
    """
    if not report_path.exists():
        raise FileNotFoundError(f"Stryker report not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid Stryker report format: expected dict, got {type(data)}")

    return data


def parse_stryker_output(
    report_path: str | None,
    working_directory: str,
) -> tuple[int, int, int, int, float, list[FileMutationStats]]:
    """
    Parse Stryker mutation testing output.

    This is the main entry point for Stryker parsing.

    Args:
        report_path: Optional explicit path to report
        working_directory: Project directory

    Returns:
        Tuple of (total_mutants, killed, survived, no_coverage, score, by_file)

    Raises:
        FileNotFoundError: If report not found
        ValueError: If report format is invalid
    """
    report_file = find_stryker_report(working_directory, report_path)

    if report_file is None:
        raise FileNotFoundError(f"Stryker report not found in expected locations. Searched in: {working_directory}")

    LOG.debug("Parsing Stryker report: %s", report_file)
    data = parse_stryker_report(report_file)

    files = _collect_mutants(data)
    by_file: list[FileMutationStats] = []
    total_killed = total_survived = total_no_coverage = total_mutants = 0

    for file_path, mutants in sorted(files.items()):
        killed, survived, no_cov = _count_mutants(mutants)
        total = len(mutants)

        total_mutants += total
        total_killed += killed
        total_survived += survived
        total_no_coverage += no_cov

        score = _compute_mutation_score(killed, survived, no_cov)

        by_file.append(
            FileMutationStats(
                filePath=str(file_path),
                totalMutants=total,
                killed=killed,
                survived=survived,
                noCoverage=no_cov,
                mutationScore=score,
            )
        )

    # Use report's score if available, otherwise compute
    score_from_report = data.get("mutationScore")
    if isinstance(score_from_report, (int, float)):
        reported_score = float(score_from_report)
        # Stryker sometimes reports as percentage (0-100)
        if reported_score > 1:
            reported_score /= 100.0
        mutation_score = reported_score
    else:
        mutation_score = _compute_mutation_score(total_killed, total_survived, total_no_coverage)

    LOG.info(
        "Stryker results: %d mutants, %d killed, %d survived, score=%.2f",
        total_mutants,
        total_killed,
        total_survived,
        mutation_score,
    )

    return (
        total_mutants,
        total_killed,
        total_survived,
        total_no_coverage,
        mutation_score,
        by_file,
    )
