"""
Mutmut mutation testing cache parser.

Mutmut is a popular Python mutation testing tool. It stores results in a
SQLite database (.mutmut-cache) rather than JSON reports.

Cache schema (mutmut 2.x):
    SourceFile: filename, hash
    Line: sourcefile_id, line, line_number
    Mutant: line_id, index, tested_against_hash, status

Status values:
    - ok_killed: Mutant was killed by tests
    - bad_survived: Mutant survived (not detected)
    - bad_timeout: Test execution timed out
    - ok_suspicious: Suspicious result
    - untested: Not yet tested
    - skipped: Skipped (pragma or config)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from models import FileMutationStats
from regions.models import Region

LOG = logging.getLogger("parsers.mutmut")


class MutmutStatus:
    """Mutmut status values (lowercase for comparison)."""

    OK_KILLED = "ok_killed"
    BAD_SURVIVED = "bad_survived"
    BAD_TIMEOUT = "bad_timeout"
    OK_SUSPICIOUS = "ok_suspicious"
    UNTESTED = "untested"
    SKIPPED = "skipped"


@dataclass
class MutmutMutant:
    """A single mutant from mutmut cache."""

    id: int
    file_path: str
    line_number: int
    status: str
    index: int  # Mutant index within line (multiple mutants per line possible)


def find_mutmut_cache(working_directory: str) -> Path | None:
    """
    Locate the mutmut cache file.

    Searches in order:
    1. .mutmut-cache in working directory
    2. .mutmut-cache in parent directories (up to 3 levels)

    Args:
        working_directory: Starting directory to search

    Returns:
        Path to cache file, or None if not found
    """
    cwd = Path(working_directory)

    # Search current and up to 3 parent directories
    for parent in [cwd] + list(cwd.parents)[:3]:
        cache_path = parent / ".mutmut-cache"
        if cache_path.exists():
            LOG.debug("Found mutmut cache: %s", cache_path)
            return cache_path

    return None


def _parse_mutmut_v2_schema(conn: sqlite3.Connection) -> list[MutmutMutant]:
    """
    Parse mutmut 2.x cache schema.

    Schema:
        mutant(id, line_id, index, tested_against_hash, status)
        line(id, sourcefile_id, line, line_number)
        sourcefile(id, filename, hash)
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            m.id,
            sf.filename,
            l.line_number,
            m.status,
            m."index"
        FROM mutant m
        JOIN line l ON m.line_id = l.id
        JOIN sourcefile sf ON l.sourcefile_id = sf.id
    """)

    mutants = []
    for row in cursor.fetchall():
        mutants.append(
            MutmutMutant(
                id=row[0],
                file_path=row[1],
                line_number=row[2],
                status=(row[3] or MutmutStatus.UNTESTED).lower(),
                index=row[4] or 0,
            )
        )

    return mutants


def _parse_mutmut_v1_schema(conn: sqlite3.Connection) -> list[MutmutMutant]:
    """
    Parse older mutmut schema (fallback).

    Some versions have a flatter schema with mutant table containing all info.
    """
    cursor = conn.cursor()

    # Try to detect available columns
    cursor.execute("PRAGMA table_info(mutant)")
    columns = {row[1] for row in cursor.fetchall()}

    if "filename" in columns and "line_number" in columns:
        # Flat schema
        cursor.execute("""
            SELECT
                id,
                filename,
                line_number,
                status,
                COALESCE(mutation_index, 0) as idx
            FROM mutant
        """)
    else:
        # Unknown schema
        return []

    mutants = []
    for row in cursor.fetchall():
        mutants.append(
            MutmutMutant(
                id=row[0],
                file_path=row[1],
                line_number=row[2],
                status=(row[3] or MutmutStatus.UNTESTED).lower(),
                index=row[4],
            )
        )

    return mutants


def parse_mutmut_cache(cache_path: Path) -> list[MutmutMutant]:
    """
    Parse mutmut SQLite cache and extract all mutants.

    Handles multiple schema versions by trying v2 first, then v1.

    Args:
        cache_path: Path to .mutmut-cache file

    Returns:
        List of all mutants in the cache

    Raises:
        FileNotFoundError: If cache doesn't exist
        sqlite3.Error: If cache is corrupted
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Mutmut cache not found: {cache_path}")

    conn = sqlite3.connect(str(cache_path))

    try:
        # Try v2 schema first (most common)
        try:
            mutants = _parse_mutmut_v2_schema(conn)
            if mutants:
                LOG.debug("Parsed %d mutants from v2 schema", len(mutants))
                return mutants
        except sqlite3.OperationalError:
            pass

        # Fall back to v1 schema
        try:
            mutants = _parse_mutmut_v1_schema(conn)
            if mutants:
                LOG.debug("Parsed %d mutants from v1 schema", len(mutants))
                return mutants
        except sqlite3.OperationalError:
            pass

        # No mutants found or unknown schema
        LOG.warning("No mutants found or unknown cache schema")
        return []

    finally:
        conn.close()


def aggregate_mutmut_stats(
    mutants: list[MutmutMutant],
) -> tuple[int, int, int, int, float, list[FileMutationStats]]:
    """
    Aggregate mutant list into summary statistics.

    Groups mutants by file and computes per-file and overall statistics.

    Args:
        mutants: List of mutants to aggregate

    Returns:
        Tuple of (total, killed, survived, no_coverage, score, by_file)
    """
    if not mutants:
        return (0, 0, 0, 0, 0.0, [])

    # Group by file
    by_file_map: dict[str, list[MutmutMutant]] = {}
    for m in mutants:
        by_file_map.setdefault(m.file_path, []).append(m)

    file_stats: list[FileMutationStats] = []
    total_killed = 0
    total_survived = 0
    total_no_coverage = 0

    for file_path in sorted(by_file_map.keys()):
        file_mutants = by_file_map[file_path]

        killed = sum(1 for m in file_mutants if m.status == MutmutStatus.OK_KILLED)
        survived = sum(1 for m in file_mutants if m.status in (MutmutStatus.BAD_SURVIVED, MutmutStatus.OK_SUSPICIOUS))
        timeout = sum(1 for m in file_mutants if m.status == MutmutStatus.BAD_TIMEOUT)
        untested = sum(1 for m in file_mutants if m.status in (MutmutStatus.UNTESTED, MutmutStatus.SKIPPED))

        total = len(file_mutants)

        # Mutation score: killed / (killed + survived)
        # Timeout and untested are excluded from score calculation
        # (they represent coverage gaps, not mutation detection)
        denominator = killed + survived
        score = killed / denominator if denominator > 0 else 0.0

        file_stats.append(
            FileMutationStats(
                filePath=file_path,
                totalMutants=total,
                killed=killed,
                survived=survived,
                noCoverage=timeout + untested,  # Combined into noCoverage
                mutationScore=score,
            )
        )

        total_killed += killed
        total_survived += survived
        total_no_coverage += timeout + untested

    total = len(mutants)
    denominator = total_killed + total_survived
    overall_score = total_killed / denominator if denominator > 0 else 0.0

    LOG.info(
        "Mutmut results: %d mutants, %d killed, %d survived, score=%.2f",
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


def parse_mutmut_output(
    working_directory: str,
    cache_path: str | None = None,
) -> tuple[int, int, int, int, float, list[FileMutationStats]]:
    """
    Parse mutmut results from cache.

    This is the main entry point for mutmut parsing, matching the signature
    of parse_stryker_output for unified interface compatibility.

    Args:
        working_directory: Project directory
        cache_path: Optional explicit path to cache file

    Returns:
        Tuple of (total, killed, survived, no_coverage, score, by_file)

    Raises:
        FileNotFoundError: If cache not found
    """
    if cache_path:
        cache = Path(cache_path)
    else:
        cache = find_mutmut_cache(working_directory)

    if cache is None or not cache.exists():
        raise FileNotFoundError(f"Mutmut cache not found. Run 'mutmut run' first. Searched in: {working_directory}")

    mutants = parse_mutmut_cache(cache)

    if not mutants:
        return (0, 0, 0, 0, 0.0, [])

    return aggregate_mutmut_stats(mutants)


def get_mutmut_region_mutants(
    working_directory: str,
    region: Region,
    cache_path: str | None = None,
) -> list[MutmutMutant]:
    """
    Get mutants within a specific region.

    Useful for region-level mutation score calculation.

    Args:
        working_directory: Project directory
        region: Region to filter to
        cache_path: Optional explicit path to cache

    Returns:
        List of mutants within the region
    """
    if cache_path:
        cache = Path(cache_path)
    else:
        cache = find_mutmut_cache(working_directory)

    if cache is None or not cache.exists():
        return []

    all_mutants = parse_mutmut_cache(cache)

    # Filter to region
    result = []
    for m in all_mutants:
        # Create a region for this mutant's location
        mutant_region = Region.for_lines(m.file_path, m.line_number, m.line_number)

        # Check if it's within the target region
        if region.contains(mutant_region) or region.overlaps(mutant_region):
            result.append(m)

    LOG.debug(
        "Found %d mutants in region %s (of %d total)",
        len(result),
        region,
        len(all_mutants),
    )

    return result


def get_region_mutation_stats(
    working_directory: str,
    region: Region,
    cache_path: str | None = None,
) -> FileMutationStats | None:
    """
    Get mutation statistics for a specific region.

    Args:
        working_directory: Project directory
        region: Region to get stats for
        cache_path: Optional explicit path to cache

    Returns:
        FileMutationStats for the region, or None if no mutants found
    """
    mutants = get_mutmut_region_mutants(working_directory, region, cache_path)

    if not mutants:
        return None

    killed = sum(1 for m in mutants if m.status == MutmutStatus.OK_KILLED)
    survived = sum(1 for m in mutants if m.status in (MutmutStatus.BAD_SURVIVED, MutmutStatus.OK_SUSPICIOUS))
    no_coverage = sum(
        1 for m in mutants if m.status in (MutmutStatus.BAD_TIMEOUT, MutmutStatus.UNTESTED, MutmutStatus.SKIPPED)
    )

    denominator = killed + survived
    score = killed / denominator if denominator > 0 else 0.0

    return FileMutationStats(
        filePath=region.to_string(),
        totalMutants=len(mutants),
        killed=killed,
        survived=survived,
        noCoverage=no_coverage,
        mutationScore=score,
    )
