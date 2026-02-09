"""
Tests for new M1 parsers: cosmic-ray, poodle, universalmutator.

Tests cover:
- cosmic-ray: JSON dump parsing, SQLite session parsing, aggregation
- poodle: JSON report parsing (mutation-testing-report-schema)
- universalmutator: killed.txt / not-killed.txt text file parsing
- Detection updates (new framework enum members, detection signals)
- Unified router integration for all new tools
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers import (
    UnsupportedFrameworkError,
    parse_mutation_output,
)
from parsers.cosmic_ray_parser import (
    _module_to_filepath,
    _normalize_test_outcome,
    _normalize_worker_outcome,
    aggregate_cosmic_ray_stats,
    find_cosmic_ray_session,
    parse_cosmic_ray_output,
    parse_cosmic_ray_session,
)
from parsers.detection import (
    MutationFramework,
    detect_available_frameworks,
)
from parsers.poodle_parser import (
    find_poodle_report,
    parse_poodle_output,
)
from parsers.universalmutator_parser import (
    _extract_source_file,
    find_universalmutator_results,
    parse_universalmutator_output,
)

# =============================================================================
# Cosmic-ray fixtures
# =============================================================================


@pytest.fixture
def cosmic_ray_json(tmp_path: Path) -> Path:
    """Create a mock cosmic-ray JSON dump."""
    data = [
        {
            "module": "mypackage.main",
            "operator": "core/NumberReplacer",
            "occurrence": 0,
            "line_number": 10,
            "job_id": "job-001",
            "test_outcome": "TestOutcome.KILLED",
            "worker_outcome": "WorkerOutcome.NORMAL",
            "diff": "--- a\n+++ b\n-x = 1\n+x = 2",
        },
        {
            "module": "mypackage.main",
            "operator": "core/NumberReplacer",
            "occurrence": 1,
            "line_number": 15,
            "job_id": "job-002",
            "test_outcome": "TestOutcome.SURVIVED",
            "worker_outcome": "WorkerOutcome.NORMAL",
            "diff": "--- a\n+++ b\n-y = 2\n+y = 3",
        },
        {
            "module": "mypackage.utils",
            "operator": "core/BooleanReplacer",
            "occurrence": 0,
            "line_number": 5,
            "job_id": "job-003",
            "test_outcome": "TestOutcome.KILLED",
            "worker_outcome": "WorkerOutcome.NORMAL",
            "diff": "",
        },
        {
            "module": "mypackage.utils",
            "operator": "core/NumberReplacer",
            "occurrence": 0,
            "line_number": 8,
            "job_id": "job-004",
            "test_outcome": "TestOutcome.SURVIVED",
            "worker_outcome": "WorkerOutcome.TIMEOUT",
            "diff": "",
        },
    ]

    report_path = tmp_path / "cosmic-ray.json"
    report_path.write_text(json.dumps(data))
    return tmp_path


@pytest.fixture
def cosmic_ray_sqlite(tmp_path: Path) -> Path:
    """Create a mock cosmic-ray SQLite session database."""
    db_path = tmp_path / "session.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE work_items (
            job_id TEXT PRIMARY KEY,
            module TEXT,
            operator_name TEXT,
            occurrence INTEGER,
            start_pos TEXT,
            end_pos TEXT,
            worker_outcome INTEGER,
            test_outcome INTEGER
        )
    """)

    # Insert test data
    items = [
        ("job-1", "pkg.module_a", "NumberReplacer", 0, "(10, 5)", "(10, 6)", 0, 1),
        ("job-2", "pkg.module_a", "NumberReplacer", 1, "(20, 3)", "(20, 4)", 0, 0),
        ("job-3", "pkg.module_b", "BooleanReplacer", 0, "(5, 1)", "(5, 5)", 0, 1),
        ("job-4", "pkg.module_b", "NumberReplacer", 0, "(8, 2)", "(8, 3)", 1, 0),  # timeout
        ("job-5", "pkg.module_a", "StringReplacer", 0, "(25, 1)", "(25, 10)", 0, 2),  # incompetent
    ]

    cursor.executemany("INSERT INTO work_items VALUES (?, ?, ?, ?, ?, ?, ?, ?)", items)
    conn.commit()
    conn.close()

    return tmp_path


@pytest.fixture
def cosmic_ray_config(tmp_path: Path) -> Path:
    """Create cosmic-ray config pointing to session file."""
    # Create a session file
    db_path = tmp_path / "my-session.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE work_items (
            job_id TEXT PRIMARY KEY,
            module TEXT,
            operator_name TEXT,
            occurrence INTEGER,
            start_pos TEXT,
            end_pos TEXT,
            worker_outcome INTEGER,
            test_outcome INTEGER
        )
    """)
    cursor.execute("INSERT INTO work_items VALUES ('j1', 'mod', 'Op', 0, '(1, 1)', '(1, 2)', 0, 1)")
    conn.commit()
    conn.close()

    # Create config file pointing to it
    config = tmp_path / ".cosmic-ray.toml"
    config.write_text('session-file = "my-session.sqlite"\n')

    return tmp_path


# =============================================================================
# Poodle fixtures
# =============================================================================


@pytest.fixture
def poodle_report(tmp_path: Path) -> Path:
    """Create a mock poodle mutation-testing-report-schema JSON report."""
    data = {
        "schemaVersion": "1",
        "thresholds": {"high": 80, "low": 60},
        "files": {
            "src/calculator.py": {
                "language": "python",
                "source": "def add(a, b): return a + b",
                "mutants": [
                    {
                        "id": "1",
                        "mutatorName": "ConditionalsBoundary",
                        "replacement": ">=",
                        "location": {
                            "start": {"line": 1, "column": 25},
                            "end": {"line": 1, "column": 26},
                        },
                        "status": "Killed",
                    },
                    {
                        "id": "2",
                        "mutatorName": "ArithmeticOperator",
                        "replacement": "-",
                        "location": {
                            "start": {"line": 1, "column": 27},
                            "end": {"line": 1, "column": 28},
                        },
                        "status": "Survived",
                    },
                    {
                        "id": "3",
                        "mutatorName": "ArithmeticOperator",
                        "replacement": "*",
                        "location": {
                            "start": {"line": 1, "column": 27},
                            "end": {"line": 1, "column": 28},
                        },
                        "status": "NoCoverage",
                    },
                ],
            },
            "src/utils.py": {
                "language": "python",
                "source": "def negate(x): return -x",
                "mutants": [
                    {
                        "id": "4",
                        "mutatorName": "UnaryOperator",
                        "replacement": "+",
                        "location": {
                            "start": {"line": 1, "column": 22},
                            "end": {"line": 1, "column": 23},
                        },
                        "status": "Killed",
                    },
                    {
                        "id": "5",
                        "mutatorName": "ReturnValue",
                        "replacement": "None",
                        "location": {
                            "start": {"line": 1, "column": 16},
                            "end": {"line": 1, "column": 24},
                        },
                        "status": "Killed",
                    },
                ],
            },
        },
    }

    report_path = tmp_path / "mutation-report.json"
    report_path.write_text(json.dumps(data))
    return tmp_path


@pytest.fixture
def poodle_report_with_score(tmp_path: Path) -> Path:
    """Create a poodle report that includes a mutationScore field."""
    data = {
        "schemaVersion": "1",
        "thresholds": {"high": 80, "low": 60},
        "mutationScore": 75.0,
        "files": {
            "src/main.py": {
                "language": "python",
                "mutants": [
                    {"id": "1", "status": "Killed"},
                    {"id": "2", "status": "Killed"},
                    {"id": "3", "status": "Killed"},
                    {"id": "4", "status": "Survived"},
                ],
            },
        },
    }

    report_path = tmp_path / "poodle-report.json"
    report_path.write_text(json.dumps(data))
    return tmp_path


# =============================================================================
# universalmutator fixtures
# =============================================================================


@pytest.fixture
def um_results(tmp_path: Path) -> Path:
    """Create mock universalmutator killed.txt and not-killed.txt."""
    killed = [
        "src/main.py.mutant.1.AOR",
        "src/main.py.mutant.2.ROR",
        "src/main.py.mutant.3.CRP",
        "src/utils.py.mutant.1.AOR",
    ]
    survived = [
        "src/main.py.mutant.4.SDL",
        "src/utils.py.mutant.2.ROR",
    ]

    (tmp_path / "killed.txt").write_text("\n".join(killed) + "\n")
    (tmp_path / "not-killed.txt").write_text("\n".join(survived) + "\n")

    return tmp_path


@pytest.fixture
def um_results_alt_name(tmp_path: Path) -> Path:
    """Create universalmutator results with alternative 'notkilled.txt' name."""
    killed = [
        "app.py.mutant.1.AOR",
        "app.py.mutant.2.ROR",
    ]
    survived = [
        "app.py.mutant.3.SDL",
    ]

    (tmp_path / "killed.txt").write_text("\n".join(killed) + "\n")
    (tmp_path / "notkilled.txt").write_text("\n".join(survived) + "\n")

    return tmp_path


@pytest.fixture
def um_results_in_subdir(tmp_path: Path) -> Path:
    """Create universalmutator results in a 'results' subdirectory."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    (results_dir / "killed.txt").write_text("mod.py.mutant.1.AOR\n")
    (results_dir / "not-killed.txt").write_text("mod.py.mutant.2.ROR\n")

    return tmp_path


# =============================================================================
# Cosmic-ray parser tests
# =============================================================================


class TestCosmicRayHelpers:
    """Tests for cosmic-ray helper functions."""

    def test_module_to_filepath(self):
        assert _module_to_filepath("mypackage.mymodule") == "mypackage/mymodule.py"
        assert _module_to_filepath("src.utils") == "src/utils.py"
        assert _module_to_filepath("main") == "main.py"

    def test_normalize_worker_outcome(self):
        assert _normalize_worker_outcome("WorkerOutcome.NORMAL") == "normal"
        assert _normalize_worker_outcome("WorkerOutcome.TIMEOUT") == "timeout"
        assert _normalize_worker_outcome("WorkerOutcome.EXCEPTION") == "exception"
        assert _normalize_worker_outcome(0) == "normal"
        assert _normalize_worker_outcome(1) == "timeout"
        assert _normalize_worker_outcome(2) == "exception"

    def test_normalize_test_outcome(self):
        assert _normalize_test_outcome("TestOutcome.KILLED") == "killed"
        assert _normalize_test_outcome("TestOutcome.SURVIVED") == "survived"
        assert _normalize_test_outcome("TestOutcome.INCOMPETENT") == "incompetent"
        assert _normalize_test_outcome(0) == "survived"
        assert _normalize_test_outcome(1) == "killed"
        assert _normalize_test_outcome(2) == "incompetent"


class TestCosmicRayFinder:
    """Tests for find_cosmic_ray_session()."""

    def test_find_json_dump(self, cosmic_ray_json: Path):
        result = find_cosmic_ray_session(str(cosmic_ray_json))
        assert result is not None
        assert result.suffix == ".json"

    def test_find_sqlite_session(self, cosmic_ray_sqlite: Path):
        result = find_cosmic_ray_session(str(cosmic_ray_sqlite))
        assert result is not None
        assert result.suffix == ".sqlite"

    def test_find_session_from_config(self, cosmic_ray_config: Path):
        result = find_cosmic_ray_session(str(cosmic_ray_config))
        assert result is not None
        assert result.name == "my-session.sqlite"

    def test_not_found(self, tmp_path: Path):
        result = find_cosmic_ray_session(str(tmp_path))
        assert result is None


class TestCosmicRayJsonParser:
    """Tests for cosmic-ray JSON dump parsing."""

    def test_parse_json_dump(self, cosmic_ray_json: Path):
        session = find_cosmic_ray_session(str(cosmic_ray_json))
        mutants = parse_cosmic_ray_session(session)

        assert len(mutants) == 4
        assert mutants[0].job_id == "job-001"
        assert mutants[0].module == "mypackage.main"
        assert mutants[0].test_outcome == "killed"
        assert mutants[0].worker_outcome == "normal"

    def test_parse_json_aggregation(self, cosmic_ray_json: Path):
        total, killed, survived, no_cov, score, by_file = parse_cosmic_ray_output(str(cosmic_ray_json))

        assert total == 4
        assert killed == 2  # job-001, job-003
        assert survived == 1  # job-002
        assert no_cov == 1  # job-004 (timeout)
        assert score == pytest.approx(2.0 / 3.0)  # 2/(2+1)
        assert len(by_file) == 2

    def test_parse_json_by_file(self, cosmic_ray_json: Path):
        _, _, _, _, _, by_file = parse_cosmic_ray_output(str(cosmic_ray_json))

        main_stats = next((f for f in by_file if "main" in f.filePath), None)
        assert main_stats is not None
        assert main_stats.totalMutants == 2
        assert main_stats.killed == 1
        assert main_stats.survived == 1


class TestCosmicRaySqliteParser:
    """Tests for cosmic-ray SQLite session parsing."""

    def test_parse_sqlite_session(self, cosmic_ray_sqlite: Path):
        session = find_cosmic_ray_session(str(cosmic_ray_sqlite))
        mutants = parse_cosmic_ray_session(session)

        assert len(mutants) == 5
        # job-1: killed/normal, job-2: survived/normal, job-3: killed/normal,
        # job-4: survived/timeout, job-5: incompetent/normal

    def test_parse_sqlite_aggregation(self, cosmic_ray_sqlite: Path):
        total, killed, survived, no_cov, score, by_file = parse_cosmic_ray_output(str(cosmic_ray_sqlite))

        assert total == 5
        assert killed == 2  # job-1, job-3
        assert survived == 1  # job-2
        assert no_cov == 2  # job-4 (timeout), job-5 (incompetent)
        assert score == pytest.approx(2.0 / 3.0)
        assert len(by_file) == 2

    def test_sqlite_line_number_parsing(self, cosmic_ray_sqlite: Path):
        """Verifies start_pos '(line, col)' format is parsed correctly."""
        session = find_cosmic_ray_session(str(cosmic_ray_sqlite))
        mutants = parse_cosmic_ray_session(session)

        job1 = next(m for m in mutants if m.job_id == "job-1")
        assert job1.line_number == 10


class TestCosmicRayEdgeCases:
    """Edge case tests for cosmic-ray parser."""

    def test_empty_json(self, tmp_path: Path):
        (tmp_path / "cosmic-ray.json").write_text("[]")
        total, killed, survived, no_cov, score, by_file = parse_cosmic_ray_output(str(tmp_path))
        assert total == 0
        assert by_file == []

    def test_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_cosmic_ray_output(str(tmp_path))

    def test_explicit_report_path(self, cosmic_ray_json: Path):
        report_file = cosmic_ray_json / "cosmic-ray.json"
        total, killed, _, _, _, _ = parse_cosmic_ray_output(str(cosmic_ray_json), report_path=str(report_file))
        assert total == 4

    def test_aggregate_empty_list(self):
        total, killed, survived, no_cov, score, by_file = aggregate_cosmic_ray_stats([])
        assert total == 0
        assert score == 0.0
        assert by_file == []


# =============================================================================
# Poodle parser tests
# =============================================================================


class TestPoodleFinder:
    """Tests for find_poodle_report()."""

    def test_find_mutation_report(self, poodle_report: Path):
        result = find_poodle_report(str(poodle_report))
        assert result is not None
        assert result.name == "mutation-report.json"

    def test_find_poodle_report_name(self, poodle_report_with_score: Path):
        result = find_poodle_report(str(poodle_report_with_score))
        assert result is not None
        assert result.name == "poodle-report.json"

    def test_not_found(self, tmp_path: Path):
        result = find_poodle_report(str(tmp_path))
        assert result is None


class TestPoodleParser:
    """Tests for poodle report parsing."""

    def test_parse_basic_report(self, poodle_report: Path):
        total, killed, survived, no_cov, score, by_file = parse_poodle_output(str(poodle_report))

        assert total == 5
        assert killed == 3  # ids 1, 4, 5
        assert survived == 1  # id 2
        assert no_cov == 1  # id 3 (NoCoverage)
        assert score == pytest.approx(3.0 / 4.0)  # 3/(3+1)
        assert len(by_file) == 2

    def test_parse_by_file(self, poodle_report: Path):
        _, _, _, _, _, by_file = parse_poodle_output(str(poodle_report))

        calc_stats = next((f for f in by_file if "calculator" in f.filePath), None)
        assert calc_stats is not None
        assert calc_stats.totalMutants == 3
        assert calc_stats.killed == 1
        assert calc_stats.survived == 1
        assert calc_stats.noCoverage == 1

        utils_stats = next((f for f in by_file if "utils" in f.filePath), None)
        assert utils_stats is not None
        assert utils_stats.totalMutants == 2
        assert utils_stats.killed == 2

    def test_report_score_used(self, poodle_report_with_score: Path):
        """When report includes mutationScore, it should be used."""
        _, _, _, _, score, _ = parse_poodle_output(str(poodle_report_with_score))
        assert score == pytest.approx(0.75)  # 75% from report

    def test_explicit_report_path(self, poodle_report: Path):
        report_file = poodle_report / "mutation-report.json"
        total, _, _, _, _, _ = parse_poodle_output(str(poodle_report), report_path=str(report_file))
        assert total == 5

    def test_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_poodle_output(str(tmp_path))


class TestPoodleEdgeCases:
    """Edge case tests for poodle parser."""

    def test_empty_files_section(self, tmp_path: Path):
        data = {"schemaVersion": "1", "files": {}}
        (tmp_path / "mutation-report.json").write_text(json.dumps(data))

        total, killed, survived, no_cov, score, by_file = parse_poodle_output(str(tmp_path))
        assert total == 0
        assert by_file == []

    def test_mixed_statuses(self, tmp_path: Path):
        """Test all poodle status values."""
        data = {
            "schemaVersion": "1",
            "files": {
                "mod.py": {
                    "mutants": [
                        {"id": "1", "status": "Killed"},
                        {"id": "2", "status": "Survived"},
                        {"id": "3", "status": "NoCoverage"},
                        {"id": "4", "status": "CompileError"},
                        {"id": "5", "status": "RuntimeError"},
                        {"id": "6", "status": "Timeout"},
                        {"id": "7", "status": "Ignored"},
                        {"id": "8", "status": "Pending"},
                    ]
                }
            },
        }
        (tmp_path / "mutation-report.json").write_text(json.dumps(data))

        total, killed, survived, no_cov, score, by_file = parse_poodle_output(str(tmp_path))
        assert total == 8
        assert killed == 1
        assert survived == 1
        assert no_cov == 1  # Only NoCoverage status
        assert score == pytest.approx(0.5)  # 1/(1+1)

    def test_invalid_json_raises(self, tmp_path: Path):
        (tmp_path / "mutation-report.json").write_text("not json")
        with pytest.raises(ValueError):  # json.JSONDecodeError wrapped as ValueError
            parse_poodle_output(str(tmp_path))


# =============================================================================
# universalmutator parser tests
# =============================================================================


class TestUniversalMutatorHelpers:
    """Tests for universalmutator helper functions."""

    def test_extract_source_standard_pattern(self):
        assert _extract_source_file("src/main.py.mutant.1.AOR") == "src/main.py"
        assert _extract_source_file("src/main.py.mutant.42.ROR") == "src/main.py"

    def test_extract_source_no_operator(self):
        assert _extract_source_file("src/main.py.mutant.1") == "src/main.py"

    def test_extract_source_alt_pattern(self):
        assert _extract_source_file("src/main.py_mutant_1") == "src/main.py"

    def test_extract_source_fallback(self):
        # Unrecognized format returns as-is
        assert _extract_source_file("unknown_format") == "unknown_format"


class TestUniversalMutatorFinder:
    """Tests for find_universalmutator_results()."""

    def test_find_in_working_dir(self, um_results: Path):
        result = find_universalmutator_results(str(um_results))
        assert result is not None
        assert result == um_results

    def test_find_in_subdir(self, um_results_in_subdir: Path):
        result = find_universalmutator_results(str(um_results_in_subdir))
        assert result is not None
        assert result.name == "results"

    def test_not_found(self, tmp_path: Path):
        result = find_universalmutator_results(str(tmp_path))
        assert result is None


class TestUniversalMutatorParser:
    """Tests for universalmutator text file parsing."""

    def test_parse_basic_results(self, um_results: Path):
        total, killed, survived, no_cov, score, by_file = parse_universalmutator_output(str(um_results))

        assert total == 6
        assert killed == 4
        assert survived == 2
        assert no_cov == 0  # universalmutator doesn't track coverage
        assert score == pytest.approx(4.0 / 6.0)
        assert len(by_file) == 2

    def test_parse_by_file(self, um_results: Path):
        _, _, _, _, _, by_file = parse_universalmutator_output(str(um_results))

        main_stats = next((f for f in by_file if "main" in f.filePath), None)
        assert main_stats is not None
        assert main_stats.totalMutants == 4  # 3 killed + 1 survived
        assert main_stats.killed == 3
        assert main_stats.survived == 1

        utils_stats = next((f for f in by_file if "utils" in f.filePath), None)
        assert utils_stats is not None
        assert utils_stats.totalMutants == 2  # 1 killed + 1 survived
        assert utils_stats.killed == 1
        assert utils_stats.survived == 1

    def test_alt_filename(self, um_results_alt_name: Path):
        """Handles 'notkilled.txt' alternative name."""
        total, killed, survived, _, score, _ = parse_universalmutator_output(str(um_results_alt_name))
        assert total == 3
        assert killed == 2
        assert survived == 1

    def test_results_in_subdir(self, um_results_in_subdir: Path):
        total, killed, survived, _, _, _ = parse_universalmutator_output(str(um_results_in_subdir))
        assert total == 2
        assert killed == 1
        assert survived == 1

    def test_explicit_report_path(self, um_results: Path):
        total, _, _, _, _, _ = parse_universalmutator_output(str(um_results), report_path=str(um_results))
        assert total == 6

    def test_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_universalmutator_output(str(tmp_path))


class TestUniversalMutatorEdgeCases:
    """Edge case tests for universalmutator parser."""

    def test_empty_files(self, tmp_path: Path):
        """Empty killed.txt and not-killed.txt should raise FileNotFoundError."""
        (tmp_path / "killed.txt").write_text("")
        (tmp_path / "not-killed.txt").write_text("")

        with pytest.raises(FileNotFoundError):
            parse_universalmutator_output(str(tmp_path))

    def test_only_killed(self, tmp_path: Path):
        """Only killed.txt present, no survived."""
        (tmp_path / "killed.txt").write_text("mod.py.mutant.1.AOR\nmod.py.mutant.2.ROR\n")
        # Create empty not-killed to prevent FileNotFoundError from both empty
        # Actually the parser should handle just killed.txt

        total, killed, survived, _, score, _ = parse_universalmutator_output(str(tmp_path))
        assert total == 2
        assert killed == 2
        assert survived == 0
        assert score == 1.0

    def test_comments_skipped(self, tmp_path: Path):
        """Comment lines in text files are skipped."""
        (tmp_path / "killed.txt").write_text("# This is a comment\nmod.py.mutant.1.AOR\n")
        (tmp_path / "not-killed.txt").write_text("# Another comment\nmod.py.mutant.2.ROR\n")

        total, killed, survived, _, _, _ = parse_universalmutator_output(str(tmp_path))
        assert total == 2
        assert killed == 1
        assert survived == 1

    def test_blank_lines_skipped(self, tmp_path: Path):
        """Blank lines in text files are skipped."""
        (tmp_path / "killed.txt").write_text("\nmod.py.mutant.1.AOR\n\n\nmod.py.mutant.2.ROR\n\n")

        total, killed, survived, _, _, _ = parse_universalmutator_output(str(tmp_path))
        assert total == 2
        assert killed == 2


# =============================================================================
# Detection updates tests
# =============================================================================


class TestDetectionUpdates:
    """Tests for new framework detection."""

    def test_poodle_enum_member_exists(self):
        assert MutationFramework.POODLE == "poodle"

    def test_universalmutator_enum_member_exists(self):
        assert MutationFramework.UNIVERSALMUTATOR == "universalmutator"

    def test_detect_poodle_from_report(self, poodle_report: Path):
        frameworks = detect_available_frameworks(str(poodle_report))
        poodle = next(
            (f for f in frameworks if f.framework == MutationFramework.POODLE),
            None,
        )
        assert poodle is not None
        assert poodle.confidence >= 0.8

    def test_detect_poodle_from_named_report(self, poodle_report_with_score: Path):
        frameworks = detect_available_frameworks(str(poodle_report_with_score))
        poodle = next(
            (f for f in frameworks if f.framework == MutationFramework.POODLE),
            None,
        )
        assert poodle is not None

    def test_detect_universalmutator(self, um_results: Path):
        frameworks = detect_available_frameworks(str(um_results))
        um = next(
            (f for f in frameworks if f.framework == MutationFramework.UNIVERSALMUTATOR),
            None,
        )
        assert um is not None
        assert um.confidence >= 0.7

    def test_detect_cosmic_ray_config(self, tmp_path: Path):
        (tmp_path / ".cosmic-ray.toml").write_text("[cosmic-ray]\n")
        frameworks = detect_available_frameworks(str(tmp_path))
        cr = next(
            (f for f in frameworks if f.framework == MutationFramework.COSMIC_RAY),
            None,
        )
        assert cr is not None

    def test_detect_poodle_from_pyproject(self, tmp_path: Path):
        """Detects poodle from [tool.poodle] in pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n\n[tool.poodle]\npaths = ["src"]\n')
        frameworks = detect_available_frameworks(str(tmp_path))
        poodle = next(
            (f for f in frameworks if f.framework == MutationFramework.POODLE),
            None,
        )
        assert poodle is not None
        assert poodle.confidence >= 0.8


# =============================================================================
# Unified router tests
# =============================================================================


class TestUnifiedRouterNewTools:
    """Tests for the unified router with new tools."""

    def test_explicit_cosmic_ray(self, cosmic_ray_json: Path):
        total, killed, _, _, _, _ = parse_mutation_output(str(cosmic_ray_json), tool="cosmic-ray")
        assert total == 4
        assert killed == 2

    def test_explicit_cosmic_ray_variation(self, cosmic_ray_json: Path):
        """Tool name variations should work."""
        for name in ["cosmic_ray", "cosmicray", "cosmic"]:
            total, _, _, _, _, _ = parse_mutation_output(str(cosmic_ray_json), tool=name)
            assert total == 4

    def test_explicit_poodle(self, poodle_report: Path):
        total, killed, _, _, _, _ = parse_mutation_output(str(poodle_report), tool="poodle")
        assert total == 5
        assert killed == 3

    def test_explicit_poodle_variation(self, poodle_report: Path):
        """poodle_test variation should work."""
        total, _, _, _, _, _ = parse_mutation_output(str(poodle_report), tool="poodle_test")
        assert total == 5

    def test_explicit_universalmutator(self, um_results: Path):
        total, killed, _, _, _, _ = parse_mutation_output(str(um_results), tool="universalmutator")
        assert total == 6
        assert killed == 4

    def test_explicit_universalmutator_variations(self, um_results: Path):
        """Tool name variations should work."""
        for name in ["universal_mutator", "um"]:
            total, _, _, _, _, _ = parse_mutation_output(str(um_results), tool=name)
            assert total == 6

    def test_cosmic_ray_no_longer_raises(self, cosmic_ray_json: Path):
        """cosmic-ray should no longer raise UnsupportedFrameworkError."""
        # This should NOT raise - it's now implemented
        total, _, _, _, _, _ = parse_mutation_output(str(cosmic_ray_json), tool="cosmic-ray")
        assert total > 0

    def test_mutpy_still_raises(self, tmp_path: Path):
        """mutpy is still not implemented."""
        with pytest.raises(UnsupportedFrameworkError, match="mutpy.*not yet implemented"):
            parse_mutation_output(str(tmp_path), tool="mutpy")

    def test_unknown_still_raises(self, tmp_path: Path):
        """Unknown tools still raise."""
        with pytest.raises(UnsupportedFrameworkError, match="Unknown mutation framework"):
            parse_mutation_output(str(tmp_path), tool="totally_fake_tool")

    def test_supported_list_updated(self, tmp_path: Path):
        """Error messages should list all supported frameworks."""
        try:
            parse_mutation_output(str(tmp_path), tool="fake")
        except UnsupportedFrameworkError as e:
            msg = str(e)
            assert "cosmic-ray" in msg
            assert "poodle" in msg
            assert "universalmutator" in msg
