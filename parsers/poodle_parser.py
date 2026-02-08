"""
Poodle (poodle-test) mutation testing parser.

Poodle is a Python mutation testing tool that outputs JSON reports following
the mutation-testing-report-schema (same schema used by Stryker).

Report format:
    {
        "schemaVersion": "1",
        "thresholds": {"high": 80, "low": 60},
        "files": {
            "src/module.py": {
                "language": "python",
                "source": "...",
                "mutants": [
                    {
                        "id": "1",
                        "mutatorName": "ConditionalsBoundary",
                        "replacement": ">=",
                        "location": {
                            "start": {"line": 10, "column": 5},
                            "end": {"line": 10, "column": 6}
                        },
                        "status": "Killed",
                        "description": "..."
                    },
                    ...
                ]
            },
            ...
        }
    }

Status values (mutation-testing-report-schema):
    - Killed: Mutant was detected by tests
    - Survived: Mutant was not detected
    - NoCoverage: No tests cover this mutant
    - CompileError: Mutant caused a compilation error
    - RuntimeError: Mutant caused a runtime error
    - Timeout: Test execution timed out
    - Ignored: Mutant was ignored (pragma, config)
    - Pending: Mutant has not been tested yet
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models import FileMutationStats

LOG = logging.getLogger("parsers.poodle")


# Status classification for mutation-testing-report-schema
_KILLED_STATUSES = {"killed"}
_SURVIVED_STATUSES = {"survived"}
_NO_COVERAGE_STATUSES = {"nocoverage", "no coverage"}
_IGNORED_STATUSES = {
    "compileerror",
    "compile error",
    "runtimeerror",
    "runtime error",
    "timeout",
    "ignored",
    "pending",
}


def find_poodle_report(working_directory: str) -> Optional[Path]:
    """
    Locate the poodle mutation testing report.

    Searches for common poodle report locations:
    1. mutation-report.json (poodle default)
    2. poodle-report.json
    3. reports/mutation/mutation.json
    4. .poodle-report.json

    Args:
        working_directory: Starting directory to search

    Returns:
        Path to report file, or None if not found
    """
    cwd = Path(working_directory)

    candidates = [
        cwd / "mutation-report.json",
        cwd / "poodle-report.json",
        cwd / ".poodle-report.json",
        cwd / "reports" / "mutation-report.json",
        cwd / "reports" / "mutation" / "mutation.json",
        cwd / "mutation-testing-report.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            LOG.debug("Found poodle report: %s", candidate)
            return candidate

    # Look for any JSON file matching mutation-testing-report schema
    for json_file in cwd.glob("*.json"):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if (
                isinstance(data, dict)
                and "files" in data
                and "schemaVersion" in data
            ):
                # Check if it looks like a mutation-testing-report
                files_val = data["files"]
                if isinstance(files_val, dict) and any(
                    isinstance(v, dict) and "mutants" in v
                    for v in files_val.values()
                ):
                    LOG.debug("Found poodle-compatible report: %s", json_file)
                    return json_file
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

    return None


def _normalize_status(status: str) -> str:
    """Normalize status string for comparison."""
    return status.lower().replace("_", "").replace(" ", "")


def _count_mutants_by_status(
    mutants: List[Dict],
) -> Tuple[int, int, int]:
    """
    Count mutants by status category.

    Returns: (killed, survived, no_coverage)
    """
    killed = 0
    survived = 0
    no_coverage = 0

    for mutant in mutants:
        status = _normalize_status(str(mutant.get("status", "")))

        if status in _KILLED_STATUSES:
            killed += 1
        elif status in _SURVIVED_STATUSES:
            survived += 1
        elif status in _NO_COVERAGE_STATUSES:
            no_coverage += 1
        # Other statuses (compileerror, timeout, ignored, pending) are
        # counted toward total but not in the score denominator

    return killed, survived, no_coverage


def parse_poodle_report(report_path: Path) -> Dict:
    """
    Parse a poodle JSON report file.

    The report follows the mutation-testing-report-schema, which is the
    same schema used by Stryker. This parser handles poodle-specific
    nuances (Python file paths, poodle status values).

    Args:
        report_path: Path to the report file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If report doesn't exist
        ValueError: If report is not valid JSON or wrong format
    """
    if not report_path.exists():
        raise FileNotFoundError(f"Poodle report not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid poodle report format: expected dict, got {type(data).__name__}"
        )

    return data


def parse_poodle_output(
    working_directory: str,
    report_path: Optional[str] = None,
) -> Tuple[int, int, int, int, float, List[FileMutationStats]]:
    """
    Parse poodle mutation testing output.

    Main entry point matching the unified parser interface (D-009).

    Args:
        working_directory: Project directory
        report_path: Optional explicit path to report file

    Returns:
        Tuple of (total, killed, survived, no_coverage, score, by_file)

    Raises:
        FileNotFoundError: If report not found
        ValueError: If report format is invalid
    """
    if report_path:
        report_file = Path(report_path)
        if not report_file.is_absolute():
            report_file_abs = Path(working_directory) / report_file
            if report_file_abs.exists():
                report_file = report_file_abs
    else:
        report_file = find_poodle_report(working_directory)

    if report_file is None or not report_file.exists():
        raise FileNotFoundError(
            f"Poodle report not found. Run 'poodle' first. "
            f"Searched in: {working_directory}"
        )

    LOG.debug("Parsing poodle report: %s", report_file)
    data = parse_poodle_report(report_file)

    # Extract files section
    files = data.get("files", {})
    if not isinstance(files, dict):
        LOG.warning("No 'files' section in poodle report")
        return (0, 0, 0, 0, 0.0, [])

    by_file: List[FileMutationStats] = []
    total_killed = 0
    total_survived = 0
    total_no_coverage = 0
    total_mutants = 0

    for file_path, file_info in sorted(files.items()):
        if not isinstance(file_info, dict):
            continue

        mutants = file_info.get("mutants", [])
        if not isinstance(mutants, list):
            continue

        killed, survived, no_cov = _count_mutants_by_status(mutants)
        total = len(mutants)

        total_mutants += total
        total_killed += killed
        total_survived += survived
        total_no_coverage += no_cov

        denominator = killed + survived
        score = killed / denominator if denominator > 0 else 0.0

        by_file.append(
            FileMutationStats(
                filePath=file_path,
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
        overall_score = float(score_from_report)
        # Normalize percentage to decimal
        if overall_score > 1:
            overall_score /= 100.0
    else:
        denominator = total_killed + total_survived
        overall_score = total_killed / denominator if denominator > 0 else 0.0

    LOG.info(
        "Poodle results: %d mutants, %d killed, %d survived, score=%.2f",
        total_mutants,
        total_killed,
        total_survived,
        overall_score,
    )

    return (
        total_mutants,
        total_killed,
        total_survived,
        total_no_coverage,
        overall_score,
        by_file,
    )
