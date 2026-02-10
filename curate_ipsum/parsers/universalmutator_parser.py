"""
universalmutator mutation testing parser.

universalmutator is a language-agnostic mutation tool that works by applying
regex-based mutations to source files. It outputs results as plain text files.

Output files:
    killed.txt     - One mutant filename per line (mutants detected by tests)
    not-killed.txt - One mutant filename per line (mutants NOT detected)
    notkilled.txt  - Alternative name for not-killed.txt

Mutant filenames typically follow the pattern:
    <original_file>.mutant.<number>.<mutation_type>
    or
    <original_file>_mutant_<number>

Examples:
    src/main.py.mutant.1.AOR
    src/utils.py.mutant.2.ROR
    src/main.py.mutant.3.CRP
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from curate_ipsum.models import FileMutationStats

LOG = logging.getLogger("parsers.universalmutator")

# Pattern to extract original file from mutant filename
# Matches: <filepath>.mutant.<number>[.<operator>]
_MUTANT_FILENAME_PATTERN = re.compile(r"^(.+?)\.mutant\.(\d+)(?:\.(.+))?$")

# Alternative pattern: <filepath>_mutant_<number>
_ALT_MUTANT_FILENAME_PATTERN = re.compile(r"^(.+?)_mutant_(\d+)$")


def find_universalmutator_results(working_directory: str) -> Path | None:
    """
    Locate universalmutator result files.

    Searches for killed.txt / not-killed.txt in common locations.

    Args:
        working_directory: Starting directory to search

    Returns:
        Path to the directory containing result files, or None
    """
    cwd = Path(working_directory)

    # Direct location
    if (cwd / "killed.txt").exists() or (cwd / "not-killed.txt").exists():
        return cwd

    if (cwd / "killed.txt").exists() or (cwd / "notkilled.txt").exists():
        return cwd

    # Check subdirectories
    for subdir_name in ("results", "mutation_results", "mutants", ".universalmutator"):
        subdir = cwd / subdir_name
        if subdir.is_dir():
            if (subdir / "killed.txt").exists() or (subdir / "not-killed.txt").exists():
                return subdir
            if (subdir / "killed.txt").exists() or (subdir / "notkilled.txt").exists():
                return subdir

    return None


def _extract_source_file(mutant_filename: str) -> str:
    """
    Extract the original source file path from a mutant filename.

    Args:
        mutant_filename: The mutant filename (e.g., "src/main.py.mutant.1.AOR")

    Returns:
        Original source file path (e.g., "src/main.py")
    """
    # Strip any directory prefix (mutant files may include paths)
    name = mutant_filename.strip()

    # Try standard pattern: file.py.mutant.N[.OP]
    match = _MUTANT_FILENAME_PATTERN.match(name)
    if match:
        return match.group(1)

    # Try alternative pattern: file.py_mutant_N
    match = _ALT_MUTANT_FILENAME_PATTERN.match(name)
    if match:
        return match.group(1)

    # Fallback: return the filename itself (can't determine source)
    return name


def _read_mutant_list(file_path: Path) -> list[str]:
    """
    Read a list of mutant filenames from a text file.

    Each line is one mutant filename. Empty lines and comment lines
    (starting with #) are skipped.

    Args:
        file_path: Path to the text file

    Returns:
        List of mutant filenames
    """
    if not file_path.exists():
        return []

    mutants = []
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                mutants.append(line)

    return mutants


def parse_universalmutator_output(
    working_directory: str,
    report_path: str | None = None,
) -> tuple[int, int, int, int, float, list[FileMutationStats]]:
    """
    Parse universalmutator results from killed.txt / not-killed.txt.

    Main entry point matching the unified parser interface (D-009).

    Args:
        working_directory: Project directory
        report_path: Optional explicit path to results directory

    Returns:
        Tuple of (total, killed, survived, no_coverage, score, by_file)

    Raises:
        FileNotFoundError: If result files not found
    """
    if report_path:
        results_dir = Path(report_path)
        if results_dir.is_file():
            results_dir = results_dir.parent
    else:
        results_dir = find_universalmutator_results(working_directory)

    if results_dir is None or not results_dir.exists():
        raise FileNotFoundError(
            f"universalmutator results not found. Run universalmutator first. "
            f"Searched for killed.txt / not-killed.txt in: {working_directory}"
        )

    # Read killed mutants
    killed_file = results_dir / "killed.txt"
    killed_mutants = _read_mutant_list(killed_file)

    # Read survived mutants (try both naming conventions)
    not_killed_file = results_dir / "not-killed.txt"
    survived_mutants = _read_mutant_list(not_killed_file)

    if not survived_mutants:
        # Try alternative name
        alt_file = results_dir / "notkilled.txt"
        survived_mutants = _read_mutant_list(alt_file)

    if not killed_mutants and not survived_mutants:
        # No results found at all
        raise FileNotFoundError(
            f"No universalmutator results found. Expected killed.txt and/or not-killed.txt in: {results_dir}"
        )

    LOG.debug(
        "Read %d killed, %d survived mutants",
        len(killed_mutants),
        len(survived_mutants),
    )

    # Group by source file
    by_file_map: dict[str, dict[str, int]] = {}  # file -> {killed, survived}

    for mutant_name in killed_mutants:
        source = _extract_source_file(mutant_name)
        if source not in by_file_map:
            by_file_map[source] = {"killed": 0, "survived": 0}
        by_file_map[source]["killed"] += 1

    for mutant_name in survived_mutants:
        source = _extract_source_file(mutant_name)
        if source not in by_file_map:
            by_file_map[source] = {"killed": 0, "survived": 0}
        by_file_map[source]["survived"] += 1

    # Build per-file stats
    file_stats: list[FileMutationStats] = []
    total_killed = 0
    total_survived = 0

    for file_path in sorted(by_file_map.keys()):
        counts = by_file_map[file_path]
        killed = counts["killed"]
        survived = counts["survived"]
        total = killed + survived

        total_killed += killed
        total_survived += survived

        denominator = killed + survived
        score = killed / denominator if denominator > 0 else 0.0

        file_stats.append(
            FileMutationStats(
                filePath=file_path,
                totalMutants=total,
                killed=killed,
                survived=survived,
                noCoverage=0,  # universalmutator doesn't track coverage
                mutationScore=score,
            )
        )

    total = total_killed + total_survived
    denominator = total_killed + total_survived
    overall_score = total_killed / denominator if denominator > 0 else 0.0

    LOG.info(
        "universalmutator results: %d mutants, %d killed, %d survived, score=%.2f",
        total,
        total_killed,
        total_survived,
        overall_score,
    )

    return (
        total,
        total_killed,
        total_survived,
        0,  # universalmutator doesn't distinguish no_coverage
        overall_score,
        file_stats,
    )
