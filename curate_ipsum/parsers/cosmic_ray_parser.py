"""
Cosmic-ray mutation testing parser.

Cosmic-ray is a Python mutation testing tool that stores results in a SQLite
session database. Results can also be exported via `cosmic-ray dump <session>`
to JSON.

Session DB schema:
    WorkItem: job_id, module, operator_name, occurrence, start_pos, end_pos,
              worker_outcome (int), test_outcome (int), diff

JSON dump format (from `cosmic-ray dump`):
    [
        {
            "module": "mypackage.mymodule",
            "operator": "core/NumberReplacer",
            "occurrence": 0,
            "line_number": 42,
            "job_id": "abc123",
            "test_outcome": "TestOutcome.KILLED",
            "worker_outcome": "WorkerOutcome.NORMAL",
            "diff": "--- a/mymodule.py\n+++ ..."
        },
        ...
    ]

Worker outcomes:
    NORMAL = 0      # Worker completed normally
    TIMEOUT = 1     # Worker timed out
    EXCEPTION = 2   # Worker raised an exception

Test outcomes:
    SURVIVED = 0    # Mutant survived (tests passed)
    KILLED = 1      # Mutant killed (tests failed)
    INCOMPETENT = 2 # Mutant was incompetent (couldn't apply)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from curate_ipsum.models import FileMutationStats

LOG = logging.getLogger("parsers.cosmic_ray")


# Worker outcome codes (from cosmic-ray's WorkerOutcome enum)
WORKER_NORMAL = 0
WORKER_TIMEOUT = 1
WORKER_EXCEPTION = 2

# Test outcome codes (from cosmic-ray's TestOutcome enum)
TEST_SURVIVED = 0
TEST_KILLED = 1
TEST_INCOMPETENT = 2


@dataclass
class CosmicRayMutant:
    """A single mutant from cosmic-ray results."""

    job_id: str
    module: str
    operator: str
    occurrence: int
    line_number: int | None
    worker_outcome: str  # "normal", "timeout", "exception"
    test_outcome: str  # "survived", "killed", "incompetent"


def find_cosmic_ray_session(working_directory: str) -> Path | None:
    """
    Locate the cosmic-ray session database or JSON dump.

    Searches for:
    1. *.sqlite session databases
    2. cosmic-ray.json dump files
    3. .cosmic-ray.toml config (to find session path)

    Args:
        working_directory: Starting directory to search

    Returns:
        Path to session/report file, or None if not found
    """
    cwd = Path(working_directory)

    # Look for JSON dump first (easiest to parse)
    json_candidates = [
        cwd / "cosmic-ray.json",
        cwd / "cosmic_ray.json",
        cwd / "cr-report.json",
    ]
    for candidate in json_candidates:
        if candidate.exists():
            LOG.debug("Found cosmic-ray JSON dump: %s", candidate)
            return candidate

    # Look for SQLite session databases
    # cosmic-ray uses .sqlite extension by convention
    sqlite_candidates = list(cwd.glob("*.sqlite"))
    if not sqlite_candidates:
        # Also check for .cosmic-ray.sqlite pattern
        sqlite_candidates = list(cwd.glob(".cosmic-ray*.sqlite"))

    if sqlite_candidates:
        # Prefer most recently modified
        best = max(sqlite_candidates, key=lambda p: p.stat().st_mtime)
        LOG.debug("Found cosmic-ray session DB: %s", best)
        return best

    # Check config file for session path
    for config_name in (".cosmic-ray.toml", "cosmic-ray.toml"):
        config_path = cwd / config_name
        if config_path.exists():
            try:
                content = config_path.read_text(encoding="utf-8")
                # Simple TOML parsing for session path
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("session-file"):
                        # session-file = "session.sqlite"
                        _, _, value = line.partition("=")
                        value = value.strip().strip("'\"")
                        session_path = cwd / value
                        if session_path.exists():
                            LOG.debug(
                                "Found cosmic-ray session from config: %s",
                                session_path,
                            )
                            return session_path
            except (OSError, UnicodeDecodeError):
                pass

    return None


def _normalize_worker_outcome(raw: object) -> str:
    """Normalize worker outcome to lowercase string."""
    s = str(raw).lower()
    if "normal" in s or s == "0":
        return "normal"
    if "timeout" in s or s == "1":
        return "timeout"
    if "exception" in s or s == "2":
        return "exception"
    return s


def _normalize_test_outcome(raw: object) -> str:
    """Normalize test outcome to lowercase string."""
    s = str(raw).lower()
    if "killed" in s or s == "1":
        return "killed"
    if "survived" in s or s == "0":
        return "survived"
    if "incompetent" in s or s == "2":
        return "incompetent"
    return s


def _module_to_filepath(module: str) -> str:
    """
    Convert Python module path to file path.

    Examples:
        "mypackage.mymodule" -> "mypackage/mymodule.py"
        "src.utils" -> "src/utils.py"
    """
    return module.replace(".", "/") + ".py"


def _parse_json_dump(json_path: Path) -> list[CosmicRayMutant]:
    """
    Parse cosmic-ray JSON dump output.

    The JSON dump is an array of objects, each representing a work item
    with its results.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        LOG.warning("Expected JSON array, got %s", type(data).__name__)
        return []

    mutants = []
    for item in data:
        if not isinstance(item, dict):
            continue

        mutants.append(
            CosmicRayMutant(
                job_id=str(item.get("job_id", "")),
                module=item.get("module", ""),
                operator=item.get("operator", ""),
                occurrence=int(item.get("occurrence", 0)),
                line_number=item.get("line_number"),
                worker_outcome=_normalize_worker_outcome(item.get("worker_outcome", "normal")),
                test_outcome=_normalize_test_outcome(item.get("test_outcome", "survived")),
            )
        )

    return mutants


def _parse_session_db(db_path: Path) -> list[CosmicRayMutant]:
    """
    Parse cosmic-ray SQLite session database.

    Schema varies across cosmic-ray versions, but the core table is
    'work_items' with columns for module, operator, outcomes, etc.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Try modern schema first (cosmic-ray >= 8.x)
        try:
            cursor.execute("""
                SELECT
                    job_id,
                    module,
                    operator_name,
                    occurrence,
                    start_pos,
                    worker_outcome,
                    test_outcome
                FROM work_items
            """)
            rows = cursor.fetchall()

            mutants = []
            for row in rows:
                job_id, module, operator, occurrence, start_pos, w_outcome, t_outcome = row

                # start_pos is typically "(line, col)" tuple stored as text or int
                line_number = None
                if start_pos is not None:
                    try:
                        if isinstance(start_pos, str) and "," in start_pos:
                            # "(line, col)" format
                            line_str = start_pos.strip("()").split(",")[0].strip()
                            line_number = int(line_str)
                        elif isinstance(start_pos, int):
                            line_number = start_pos
                    except (ValueError, IndexError):
                        pass

                mutants.append(
                    CosmicRayMutant(
                        job_id=str(job_id or ""),
                        module=module or "",
                        operator=operator or "",
                        occurrence=int(occurrence or 0),
                        line_number=line_number,
                        worker_outcome=_normalize_worker_outcome(w_outcome),
                        test_outcome=_normalize_test_outcome(t_outcome),
                    )
                )

            return mutants

        except sqlite3.OperationalError:
            pass

        # Try older schema (cosmic-ray < 8.x)
        try:
            cursor.execute("""
                SELECT
                    job_id,
                    module_path,
                    operator,
                    occurrence,
                    line_number,
                    worker_outcome,
                    test_outcome
                FROM mutation_specs
                LEFT JOIN results ON mutation_specs.job_id = results.job_id
            """)
            rows = cursor.fetchall()

            mutants = []
            for row in rows:
                mutants.append(
                    CosmicRayMutant(
                        job_id=str(row[0] or ""),
                        module=row[1] or "",
                        operator=row[2] or "",
                        occurrence=int(row[3] or 0),
                        line_number=row[4],
                        worker_outcome=_normalize_worker_outcome(row[5]),
                        test_outcome=_normalize_test_outcome(row[6]),
                    )
                )

            return mutants

        except sqlite3.OperationalError:
            pass

        LOG.warning("Unknown cosmic-ray session DB schema")
        return []

    finally:
        conn.close()


def parse_cosmic_ray_session(session_path: Path) -> list[CosmicRayMutant]:
    """
    Parse cosmic-ray results from session file (SQLite or JSON).

    Auto-detects format based on file content.

    Args:
        session_path: Path to session DB or JSON dump

    Returns:
        List of all mutants

    Raises:
        FileNotFoundError: If session file doesn't exist
    """
    if not session_path.exists():
        raise FileNotFoundError(f"Cosmic-ray session not found: {session_path}")

    # Detect format: try JSON first (faster check), then SQLite
    suffix = session_path.suffix.lower()
    if suffix == ".json":
        return _parse_json_dump(session_path)

    if suffix in (".sqlite", ".db"):
        return _parse_session_db(session_path)

    # Ambiguous extension - try JSON first, then SQLite
    try:
        return _parse_json_dump(session_path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    try:
        return _parse_session_db(session_path)
    except sqlite3.DatabaseError:
        pass

    LOG.warning("Could not parse cosmic-ray session: %s", session_path)
    return []


def aggregate_cosmic_ray_stats(
    mutants: list[CosmicRayMutant],
) -> tuple[int, int, int, int, float, list[FileMutationStats]]:
    """
    Aggregate cosmic-ray mutants into summary statistics.

    Groups mutants by module (converted to file path) and computes
    per-file and overall statistics.

    Classification:
    - killed: test_outcome == "killed" AND worker_outcome == "normal"
    - survived: test_outcome == "survived" AND worker_outcome == "normal"
    - no_coverage: worker_outcome in ("timeout", "exception")
                   OR test_outcome == "incompetent"

    Args:
        mutants: List of mutants to aggregate

    Returns:
        Tuple of (total, killed, survived, no_coverage, score, by_file)
    """
    if not mutants:
        return (0, 0, 0, 0, 0.0, [])

    # Group by module -> file path
    by_file_map: dict[str, list[CosmicRayMutant]] = {}
    for m in mutants:
        file_path = _module_to_filepath(m.module) if m.module else "<unknown>"
        by_file_map.setdefault(file_path, []).append(m)

    file_stats: list[FileMutationStats] = []
    total_killed = 0
    total_survived = 0
    total_no_coverage = 0

    for file_path in sorted(by_file_map.keys()):
        file_mutants = by_file_map[file_path]

        killed = sum(1 for m in file_mutants if m.test_outcome == "killed" and m.worker_outcome == "normal")
        survived = sum(1 for m in file_mutants if m.test_outcome == "survived" and m.worker_outcome == "normal")
        no_coverage = sum(
            1 for m in file_mutants if m.worker_outcome in ("timeout", "exception") or m.test_outcome == "incompetent"
        )

        total = len(file_mutants)
        denominator = killed + survived
        score = killed / denominator if denominator > 0 else 0.0

        file_stats.append(
            FileMutationStats(
                filePath=file_path,
                totalMutants=total,
                killed=killed,
                survived=survived,
                noCoverage=no_coverage,
                mutationScore=score,
            )
        )

        total_killed += killed
        total_survived += survived
        total_no_coverage += no_coverage

    total = len(mutants)
    denominator = total_killed + total_survived
    overall_score = total_killed / denominator if denominator > 0 else 0.0

    LOG.info(
        "Cosmic-ray results: %d mutants, %d killed, %d survived, score=%.2f",
        total,
        total_killed,
        total_survived,
        overall_score,
    )

    return (
        total,
        total_killed,
        total_survived,
        total_no_coverage,
        overall_score,
        file_stats,
    )


def parse_cosmic_ray_output(
    working_directory: str,
    report_path: str | None = None,
) -> tuple[int, int, int, int, float, list[FileMutationStats]]:
    """
    Parse cosmic-ray mutation testing output.

    Main entry point matching the unified parser interface (D-009).

    Args:
        working_directory: Project directory
        report_path: Optional explicit path to session file or JSON dump

    Returns:
        Tuple of (total, killed, survived, no_coverage, score, by_file)

    Raises:
        FileNotFoundError: If session/report not found
    """
    if report_path:
        session = Path(report_path)
    else:
        session = find_cosmic_ray_session(working_directory)

    if session is None or not session.exists():
        raise FileNotFoundError(
            f"Cosmic-ray session not found. Run 'cosmic-ray init <config> <session>' "
            f"and 'cosmic-ray exec <session>' first. "
            f"Searched in: {working_directory}"
        )

    mutants = parse_cosmic_ray_session(session)

    if not mutants:
        return (0, 0, 0, 0, 0.0, [])

    return aggregate_cosmic_ray_stats(mutants)
