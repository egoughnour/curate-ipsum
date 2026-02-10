"""
Tests for mutation testing parsers (M1: Multi-Framework Foundation).

Tests cover:
- Framework detection
- Language detection
- Stryker parser
- Mutmut parser
- Unified parser interface
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from curate_ipsum.parsers import (
    UnsupportedFrameworkError,
    parse_mutation_output,
)
from curate_ipsum.parsers.detection import (
    MutationFramework,
    detect_available_frameworks,
    detect_language,
    recommend_framework,
)
from curate_ipsum.parsers.mutmut_parser import (
    find_mutmut_cache,
    parse_mutmut_output,
)
from curate_ipsum.parsers.stryker_parser import (
    find_stryker_report,
    parse_stryker_output,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a mock Python project structure."""
    # Create Python files
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text("def main(): pass")
    (src / "utils.py").write_text("def helper(): pass")

    # Create config files
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')
    (tmp_path / "requirements.txt").write_text("pytest\n")

    return tmp_path


@pytest.fixture
def js_project(tmp_path: Path) -> Path:
    """Create a mock JavaScript project structure."""
    # Create JS files
    src = tmp_path / "src"
    src.mkdir()
    (src / "index.js").write_text("export default {}")
    (src / "utils.js").write_text("export function helper() {}")

    # Create config files
    (tmp_path / "package.json").write_text('{"name": "test"}')

    return tmp_path


@pytest.fixture
def stryker_report(tmp_path: Path) -> Path:
    """Create a mock Stryker report."""
    reports_dir = tmp_path / "reports" / "mutation"
    reports_dir.mkdir(parents=True)

    report_data = {
        "schemaVersion": "1.0",
        "thresholds": {"high": 80, "low": 60},
        "files": {
            "src/index.js": {
                "language": "javascript",
                "mutants": [
                    {"id": "1", "status": "Killed", "location": {"start": {"line": 1}}},
                    {"id": "2", "status": "Killed", "location": {"start": {"line": 2}}},
                    {"id": "3", "status": "Survived", "location": {"start": {"line": 3}}},
                    {"id": "4", "status": "NoCoverage", "location": {"start": {"line": 4}}},
                ],
            },
            "src/utils.js": {
                "language": "javascript",
                "mutants": [
                    {"id": "5", "status": "Killed", "location": {"start": {"line": 1}}},
                    {"id": "6", "status": "Killed", "location": {"start": {"line": 2}}},
                ],
            },
        },
        "mutationScore": 66.67,
    }

    report_path = reports_dir / "mutation.json"
    report_path.write_text(json.dumps(report_data))

    return tmp_path


@pytest.fixture
def mutmut_cache(tmp_path: Path) -> Path:
    """Create a mock mutmut cache database."""
    cache_path = tmp_path / ".mutmut-cache"

    conn = sqlite3.connect(str(cache_path))
    cursor = conn.cursor()

    # Create mutmut v2 schema
    cursor.execute("""
        CREATE TABLE sourcefile (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            hash TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE line (
            id INTEGER PRIMARY KEY,
            sourcefile_id INTEGER,
            line TEXT,
            line_number INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE mutant (
            id INTEGER PRIMARY KEY,
            line_id INTEGER,
            "index" INTEGER,
            tested_against_hash TEXT,
            status TEXT
        )
    """)

    # Insert test data
    cursor.execute("INSERT INTO sourcefile VALUES (1, 'src/main.py', 'abc123')")
    cursor.execute("INSERT INTO sourcefile VALUES (2, 'src/utils.py', 'def456')")

    cursor.execute("INSERT INTO line VALUES (1, 1, 'x = 1', 10)")
    cursor.execute("INSERT INTO line VALUES (2, 1, 'y = 2', 15)")
    cursor.execute("INSERT INTO line VALUES (3, 2, 'z = 3', 5)")

    cursor.execute("INSERT INTO mutant VALUES (1, 1, 0, 'hash1', 'ok_killed')")
    cursor.execute("INSERT INTO mutant VALUES (2, 1, 1, 'hash1', 'ok_killed')")
    cursor.execute("INSERT INTO mutant VALUES (3, 2, 0, 'hash1', 'bad_survived')")
    cursor.execute("INSERT INTO mutant VALUES (4, 3, 0, 'hash1', 'ok_killed')")
    cursor.execute("INSERT INTO mutant VALUES (5, 3, 1, 'hash1', 'bad_timeout')")

    conn.commit()
    conn.close()

    return tmp_path


# =============================================================================
# Language Detection Tests
# =============================================================================


class TestLanguageDetection:
    """Tests for detect_language()."""

    def test_detect_python_project(self, python_project: Path):
        """Detects Python projects correctly."""
        result = detect_language(str(python_project))

        assert result.primary == "python"
        assert result.confidence > 0.5

    def test_detect_javascript_project(self, js_project: Path):
        """Detects JavaScript projects correctly."""
        result = detect_language(str(js_project))

        assert result.primary == "javascript"
        assert result.confidence > 0.5

    def test_config_files_boost_confidence(self, tmp_path: Path):
        """Config files increase language detection confidence."""
        # Create pyproject.toml without any .py files
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')

        result = detect_language(str(tmp_path))

        assert result.primary == "python"
        assert result.confidence > 0

    def test_empty_directory(self, tmp_path: Path):
        """Empty directories return unknown."""
        result = detect_language(str(tmp_path))

        assert result.primary == "unknown"
        assert result.confidence == 0.0

    def test_nonexistent_directory(self, tmp_path: Path):
        """Nonexistent directories return unknown."""
        result = detect_language(str(tmp_path / "nonexistent"))

        assert result.primary == "unknown"


# =============================================================================
# Framework Detection Tests
# =============================================================================


class TestFrameworkDetection:
    """Tests for detect_available_frameworks()."""

    def test_detect_stryker_from_report(self, stryker_report: Path):
        """Detects Stryker from report file."""
        frameworks = detect_available_frameworks(str(stryker_report))

        assert len(frameworks) >= 1
        stryker = next((f for f in frameworks if f.framework == MutationFramework.STRYKER), None)
        assert stryker is not None
        assert stryker.confidence > 0.9

    def test_detect_stryker_from_config(self, tmp_path: Path):
        """Detects Stryker from config file."""
        (tmp_path / "stryker.conf.js").write_text("module.exports = {}")

        frameworks = detect_available_frameworks(str(tmp_path))

        stryker = next((f for f in frameworks if f.framework == MutationFramework.STRYKER), None)
        assert stryker is not None
        assert stryker.confidence > 0.7

    def test_detect_mutmut_from_cache(self, mutmut_cache: Path):
        """Detects mutmut from cache file."""
        frameworks = detect_available_frameworks(str(mutmut_cache))

        mutmut = next((f for f in frameworks if f.framework == MutationFramework.MUTMUT), None)
        assert mutmut is not None
        assert mutmut.confidence > 0.9

    def test_detect_mutmut_from_setup_cfg(self, tmp_path: Path):
        """Detects mutmut from setup.cfg [mutmut] section."""
        (tmp_path / "setup.cfg").write_text("[mutmut]\npaths_to_mutate=src/")

        frameworks = detect_available_frameworks(str(tmp_path))

        mutmut = next((f for f in frameworks if f.framework == MutationFramework.MUTMUT), None)
        assert mutmut is not None

    def test_no_frameworks_detected(self, tmp_path: Path):
        """Returns empty list when no frameworks detected."""
        frameworks = detect_available_frameworks(str(tmp_path))

        assert frameworks == []


class TestFrameworkRecommendation:
    """Tests for recommend_framework()."""

    def test_recommend_detected_framework(self, stryker_report: Path):
        """Recommends detected framework over language-based."""
        result = recommend_framework(str(stryker_report))

        assert result.framework == MutationFramework.STRYKER
        assert result.confidence > 0.9

    def test_recommend_mutmut_for_python(self, python_project: Path):
        """Recommends mutmut for Python projects without existing setup."""
        result = recommend_framework(str(python_project))

        assert result.framework == MutationFramework.MUTMUT

    def test_recommend_stryker_for_javascript(self, js_project: Path):
        """Recommends Stryker for JavaScript projects without existing setup."""
        result = recommend_framework(str(js_project))

        assert result.framework == MutationFramework.STRYKER

    def test_recommend_unknown_for_empty(self, tmp_path: Path):
        """Returns UNKNOWN for projects without detectable language."""
        result = recommend_framework(str(tmp_path))

        assert result.framework == MutationFramework.UNKNOWN


# =============================================================================
# Stryker Parser Tests
# =============================================================================


class TestStrykerParser:
    """Tests for Stryker report parsing."""

    def test_find_stryker_report(self, stryker_report: Path):
        """Finds Stryker report in default location."""
        result = find_stryker_report(str(stryker_report))

        assert result is not None
        assert result.exists()
        assert result.name == "mutation.json"

    def test_find_stryker_report_not_found(self, tmp_path: Path):
        """Returns None when report not found."""
        result = find_stryker_report(str(tmp_path))

        assert result is None

    def test_parse_stryker_output(self, stryker_report: Path):
        """Parses Stryker report correctly."""
        total, killed, survived, no_cov, score, by_file = parse_stryker_output(None, str(stryker_report))

        assert total == 6
        assert killed == 4
        assert survived == 1
        assert no_cov == 1
        assert 0.6 < score < 0.7  # ~66.67%
        assert len(by_file) == 2

    def test_parse_stryker_by_file(self, stryker_report: Path):
        """Parses per-file statistics correctly."""
        _, _, _, _, _, by_file = parse_stryker_output(None, str(stryker_report))

        # Find src/index.js
        index_stats = next((f for f in by_file if "index" in f.filePath), None)
        assert index_stats is not None
        assert index_stats.totalMutants == 4
        assert index_stats.killed == 2

    def test_parse_stryker_not_found(self, tmp_path: Path):
        """Raises FileNotFoundError when report not found."""
        with pytest.raises(FileNotFoundError):
            parse_stryker_output(None, str(tmp_path))


# =============================================================================
# Mutmut Parser Tests
# =============================================================================


class TestMutmutParser:
    """Tests for mutmut cache parsing."""

    def test_find_mutmut_cache(self, mutmut_cache: Path):
        """Finds mutmut cache in working directory."""
        result = find_mutmut_cache(str(mutmut_cache))

        assert result is not None
        assert result.exists()
        assert result.name == ".mutmut-cache"

    def test_find_mutmut_cache_not_found(self, tmp_path: Path):
        """Returns None when cache not found."""
        result = find_mutmut_cache(str(tmp_path))

        assert result is None

    def test_parse_mutmut_output(self, mutmut_cache: Path):
        """Parses mutmut cache correctly."""
        total, killed, survived, no_cov, score, by_file = parse_mutmut_output(str(mutmut_cache))

        assert total == 5
        assert killed == 3
        assert survived == 1
        assert no_cov == 1  # timeout counts as no_coverage
        assert score == 0.75  # 3/(3+1)
        assert len(by_file) == 2

    def test_parse_mutmut_by_file(self, mutmut_cache: Path):
        """Parses per-file statistics correctly."""
        _, _, _, _, _, by_file = parse_mutmut_output(str(mutmut_cache))

        # Find src/main.py
        main_stats = next((f for f in by_file if "main" in f.filePath), None)
        assert main_stats is not None
        assert main_stats.totalMutants == 3  # 2 on line 10, 1 on line 15
        assert main_stats.killed == 2
        assert main_stats.survived == 1

    def test_parse_mutmut_not_found(self, tmp_path: Path):
        """Raises FileNotFoundError when cache not found."""
        with pytest.raises(FileNotFoundError):
            parse_mutmut_output(str(tmp_path))


# =============================================================================
# Unified Parser Tests
# =============================================================================


class TestUnifiedParser:
    """Tests for the unified parse_mutation_output() interface."""

    def test_explicit_stryker(self, stryker_report: Path):
        """Parses Stryker when explicitly specified."""
        total, killed, survived, _, score, _ = parse_mutation_output(str(stryker_report), tool="stryker")

        assert total == 6
        assert killed == 4

    def test_explicit_mutmut(self, mutmut_cache: Path):
        """Parses mutmut when explicitly specified."""
        total, killed, survived, _, score, _ = parse_mutation_output(str(mutmut_cache), tool="mutmut")

        assert total == 5
        assert killed == 3

    def test_auto_detect_stryker(self, stryker_report: Path):
        """Auto-detects and parses Stryker."""
        total, killed, _, _, _, _ = parse_mutation_output(str(stryker_report))

        assert total == 6
        assert killed == 4

    def test_auto_detect_mutmut(self, mutmut_cache: Path):
        """Auto-detects and parses mutmut."""
        # Add a Python file so language detection works
        (mutmut_cache / "main.py").write_text("x = 1")

        total, killed, _, _, _, _ = parse_mutation_output(str(mutmut_cache))

        assert total == 5
        assert killed == 3

    def test_unsupported_framework(self, tmp_path: Path):
        """Raises UnsupportedFrameworkError for unknown frameworks."""
        with pytest.raises(UnsupportedFrameworkError, match="Unknown mutation framework"):
            parse_mutation_output(str(tmp_path), tool="unknown_tool")

    def test_cosmic_ray_no_session_raises_file_not_found(self, tmp_path: Path):
        """cosmic-ray is now implemented; raises FileNotFoundError if no session found."""
        with pytest.raises(FileNotFoundError, match="Cosmic-ray session not found"):
            parse_mutation_output(str(tmp_path), tool="cosmic-ray")

    def test_tool_name_normalization(self, stryker_report: Path):
        """Tool names are normalized (case insensitive, handle variations)."""
        # These should all work
        for tool_name in ["Stryker", "STRYKER", "stryker_js"]:
            total, _, _, _, _, _ = parse_mutation_output(str(stryker_report), tool=tool_name)
            assert total == 6
