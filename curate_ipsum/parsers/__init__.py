"""
Unified mutation testing parser interface.

This module provides a single entry point for parsing mutation testing output
from multiple frameworks (Stryker, mutmut, etc.) with automatic framework
detection.

Usage:
    from curate_ipsum.parsers import parse_mutation_output

    # Auto-detect framework
    result = parse_mutation_output(working_directory="/path/to/project")

    # Explicit framework
    result = parse_mutation_output(
        working_directory="/path/to/project",
        tool="mutmut"
    )
"""

from __future__ import annotations

import logging

from curate_ipsum.models import FileMutationStats
from curate_ipsum.parsers.cosmic_ray_parser import parse_cosmic_ray_output
from curate_ipsum.parsers.detection import (
    FrameworkDetection,
    MutationFramework,
    ProjectLanguage,
    detect_available_frameworks,
    detect_language,
    recommend_framework,
)
from curate_ipsum.parsers.mutmut_parser import parse_mutmut_output
from curate_ipsum.parsers.poodle_parser import parse_poodle_output
from curate_ipsum.parsers.stryker_parser import parse_stryker_output
from curate_ipsum.parsers.universalmutator_parser import parse_universalmutator_output

LOG = logging.getLogger("parsers")

__all__ = [
    # Main interface
    "parse_mutation_output",
    "UnsupportedFrameworkError",
    # Detection
    "MutationFramework",
    "FrameworkDetection",
    "ProjectLanguage",
    "detect_language",
    "detect_available_frameworks",
    "recommend_framework",
    # Individual parsers (for direct use)
    "parse_stryker_output",
    "parse_mutmut_output",
    "parse_cosmic_ray_output",
    "parse_poodle_output",
    "parse_universalmutator_output",
]


class UnsupportedFrameworkError(Exception):
    """Raised when a mutation framework is not supported."""

    pass


def parse_mutation_output(
    working_directory: str,
    tool: str | None = None,
    report_path: str | None = None,
) -> tuple[int, int, int, int, float, list[FileMutationStats]]:
    """
    Parse mutation testing output, auto-detecting framework if not specified.

    This is the main entry point for all mutation testing parsers. It routes
    to the appropriate parser based on the tool parameter or auto-detection.

    Args:
        working_directory: Project directory containing mutation output
        tool: Optional framework name ("stryker", "mutmut", etc.)
              If None, auto-detects based on available output files
        report_path: Optional path to report file or cache
                    (interpretation depends on framework)

    Returns:
        Tuple of (total_mutants, killed, survived, no_coverage, score, by_file)

        - total_mutants: Total number of mutants processed
        - killed: Number of mutants killed by tests
        - survived: Number of mutants that survived
        - no_coverage: Number of mutants with no test coverage
        - score: Mutation score as killed/(killed+survived)
        - by_file: Per-file breakdown as FileMutationStats list

    Raises:
        UnsupportedFrameworkError: If framework is not supported
        FileNotFoundError: If report/cache not found

    Examples:
        # Auto-detect framework
        >>> total, killed, survived, no_cov, score, by_file = parse_mutation_output("./")

        # Explicit Stryker
        >>> parse_mutation_output("./", tool="stryker")

        # Explicit mutmut with custom cache path
        >>> parse_mutation_output("./", tool="mutmut", report_path="./.mutmut-cache")
    """
    # Auto-detect if not specified
    if tool is None:
        detection = recommend_framework(working_directory)
        tool = detection.framework.value
        LOG.info(
            "Auto-detected framework: %s (confidence=%.2f, reason=%s)",
            tool,
            detection.confidence,
            detection.evidence,
        )

        if detection.framework == MutationFramework.UNKNOWN:
            raise UnsupportedFrameworkError(
                f"Could not auto-detect mutation framework. {detection.evidence}. "
                f"Supported frameworks: stryker, mutmut, cosmic-ray, poodle, universalmutator"
            )

    tool_lower = tool.lower().replace("-", "_").replace(" ", "_")

    # Route to appropriate parser
    if tool_lower in ("stryker", "stryker_js"):
        return parse_stryker_output(report_path, working_directory)

    elif tool_lower in ("mutmut", "mut_mut"):
        return parse_mutmut_output(working_directory, report_path)

    elif tool_lower in ("cosmic_ray", "cosmicray", "cosmic"):
        return parse_cosmic_ray_output(working_directory, report_path)

    elif tool_lower in ("poodle", "poodle_test"):
        return parse_poodle_output(working_directory, report_path)

    elif tool_lower in ("universalmutator", "universal_mutator", "um"):
        return parse_universalmutator_output(working_directory, report_path)

    elif tool_lower in ("mutpy", "mut_py"):
        raise UnsupportedFrameworkError(
            "mutpy parser not yet implemented. "
            "Supported frameworks: stryker, mutmut, cosmic-ray, poodle, universalmutator"
        )

    else:
        raise UnsupportedFrameworkError(
            f"Unknown mutation framework: {tool}. "
            f"Supported frameworks: stryker, mutmut, cosmic-ray, poodle, universalmutator"
        )


def get_detected_tool(working_directory: str) -> str | None:
    """
    Get the name of the auto-detected tool without parsing.

    Useful when you need to know which tool was detected before
    running the full parse.

    Args:
        working_directory: Project directory

    Returns:
        Tool name string, or None if no tool detected
    """
    detection = recommend_framework(working_directory)

    if detection.framework == MutationFramework.UNKNOWN:
        return None

    return detection.framework.value
