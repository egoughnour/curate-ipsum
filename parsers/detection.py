"""
Mutation framework and language detection.

Detects:
- Project language (Python, JavaScript, etc.)
- Available/configured mutation frameworks
- Recommends the best framework for a project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

LOG = logging.getLogger("parsers.detection")


class MutationFramework(str, Enum):
    """Supported mutation testing frameworks."""

    STRYKER = "stryker"
    MUTMUT = "mutmut"
    COSMIC_RAY = "cosmic-ray"
    MUTPY = "mutpy"
    UNKNOWN = "unknown"


@dataclass
class FrameworkDetection:
    """Result of framework detection."""

    framework: MutationFramework
    confidence: float  # 0.0 to 1.0
    evidence: str  # What triggered detection


@dataclass
class ProjectLanguage:
    """Detected project language(s)."""

    primary: str
    secondary: List[str]
    confidence: float


# Language detection: file extension to language mapping
_EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".kt": "kotlin",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".php": "php",
    ".scala": "scala",
}

# Config files that strongly indicate language
_CONFIG_LANGUAGE_SIGNALS: Dict[str, str] = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "requirements.txt": "python",
    "Pipfile": "python",
    "poetry.lock": "python",
    "package.json": "javascript",
    "tsconfig.json": "typescript",
    "package-lock.json": "javascript",
    "yarn.lock": "javascript",
    "pnpm-lock.yaml": "javascript",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "go.sum": "go",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "kotlin",
    "Gemfile": "ruby",
    "Gemfile.lock": "ruby",
    "composer.json": "php",
    "build.sbt": "scala",
    "Package.swift": "swift",
}

# Directories to skip when scanning for files
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    "target",
    ".idea",
    ".vscode",
}


def detect_language(working_directory: str) -> ProjectLanguage:
    """
    Detect primary language of a project.

    Examines file extensions and configuration files to determine
    the project's primary programming language.

    Args:
        working_directory: Project directory to analyze

    Returns:
        ProjectLanguage with primary language and confidence
    """
    cwd = Path(working_directory)

    if not cwd.exists():
        return ProjectLanguage(primary="unknown", secondary=[], confidence=0.0)

    # Count files by extension (limited depth to avoid scanning huge dirs)
    ext_counts: Dict[str, int] = {}
    max_files = 1000  # Limit to avoid performance issues
    files_scanned = 0

    for f in cwd.rglob("*"):
        if files_scanned >= max_files:
            break

        # Skip hidden and common non-source directories
        if any(part in _SKIP_DIRS for part in f.parts):
            continue

        if f.is_file():
            files_scanned += 1
            ext = f.suffix.lower()
            if ext in _EXTENSION_LANGUAGE_MAP:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # Score languages from file counts
    lang_scores: Dict[str, float] = {}
    for ext, count in ext_counts.items():
        lang = _EXTENSION_LANGUAGE_MAP.get(ext)
        if lang:
            lang_scores[lang] = lang_scores.get(lang, 0) + count

    # Boost from config files (strong signal)
    config_weight = 50  # Each config file counts as 50 source files
    for config, lang in _CONFIG_LANGUAGE_SIGNALS.items():
        if (cwd / config).exists():
            lang_scores[lang] = lang_scores.get(lang, 0) + config_weight
            LOG.debug("Found config file %s -> %s", config, lang)

    if not lang_scores:
        return ProjectLanguage(primary="unknown", secondary=[], confidence=0.0)

    # Sort by score descending
    sorted_langs = sorted(lang_scores.items(), key=lambda x: -x[1])
    total = sum(lang_scores.values())

    primary = sorted_langs[0][0]
    primary_score = sorted_langs[0][1]
    confidence = min(1.0, primary_score / total) if total > 0 else 0.0

    # Secondary languages (up to 3)
    secondary = [lang for lang, _ in sorted_langs[1:4] if lang != primary]

    LOG.info(
        "Detected language: %s (confidence=%.2f, secondary=%s)",
        primary,
        confidence,
        secondary,
    )

    return ProjectLanguage(
        primary=primary,
        secondary=secondary,
        confidence=confidence,
    )


def detect_available_frameworks(working_directory: str) -> List[FrameworkDetection]:
    """
    Detect which mutation frameworks have been run or are configured.

    Checks for:
    - Output files/directories from each framework
    - Configuration files
    - Cache files

    Args:
        working_directory: Project directory to analyze

    Returns:
        List of detected frameworks with confidence scores
    """
    cwd = Path(working_directory)
    detections: List[FrameworkDetection] = []

    # Stryker detection (JavaScript/TypeScript)
    stryker_signals = [
        (cwd / "stryker.conf.js", 0.8, "config"),
        (cwd / "stryker.conf.json", 0.8, "config"),
        (cwd / "stryker.conf.mjs", 0.8, "config"),
        (cwd / ".stryker-tmp", 0.7, "temp directory"),
        (cwd / "reports" / "mutation" / "mutation.json", 0.95, "report"),
        (cwd / "reports" / "stryker-report.json", 0.95, "report"),
    ]
    for signal_path, confidence, evidence_type in stryker_signals:
        if signal_path.exists():
            detections.append(
                FrameworkDetection(
                    framework=MutationFramework.STRYKER,
                    confidence=confidence,
                    evidence=f"Found {evidence_type}: {signal_path.name}",
                )
            )
            break

    # Mutmut detection (Python)
    mutmut_signals = [
        (cwd / ".mutmut-cache", 0.95, "cache database"),
        (cwd / "mutmut.toml", 0.8, "config"),
    ]
    for signal_path, confidence, evidence_type in mutmut_signals:
        if signal_path.exists():
            detections.append(
                FrameworkDetection(
                    framework=MutationFramework.MUTMUT,
                    confidence=confidence,
                    evidence=f"Found {evidence_type}: {signal_path.name}",
                )
            )
            break

    # Check setup.cfg for [mutmut] section
    setup_cfg = cwd / "setup.cfg"
    if setup_cfg.exists():
        try:
            content = setup_cfg.read_text(encoding="utf-8")
            if "[mutmut]" in content:
                detections.append(
                    FrameworkDetection(
                        framework=MutationFramework.MUTMUT,
                        confidence=0.7,
                        evidence="Found [mutmut] section in setup.cfg",
                    )
                )
        except (OSError, UnicodeDecodeError):
            pass

    # Cosmic-ray detection (Python)
    cosmic_signals = [
        (cwd / ".cosmic-ray.toml", 0.8, "config"),
        (cwd / "cosmic-ray.toml", 0.8, "config"),
    ]
    for signal_path, confidence, evidence_type in cosmic_signals:
        if signal_path.exists():
            detections.append(
                FrameworkDetection(
                    framework=MutationFramework.COSMIC_RAY,
                    confidence=confidence,
                    evidence=f"Found {evidence_type}: {signal_path.name}",
                )
            )
            break

    # MutPy detection (Python) - less common
    if (cwd / ".mutpy").exists():
        detections.append(
            FrameworkDetection(
                framework=MutationFramework.MUTPY,
                confidence=0.7,
                evidence="Found .mutpy directory",
            )
        )

    LOG.info("Detected %d mutation framework(s)", len(detections))
    return detections


def recommend_framework(working_directory: str) -> FrameworkDetection:
    """
    Recommend the best mutation framework for a project.

    Priority:
    1. Already-run frameworks (have output/cache)
    2. Configured frameworks
    3. Language-based recommendation

    Args:
        working_directory: Project directory

    Returns:
        Recommended framework with confidence and reasoning
    """
    # Check what's already been run or configured
    detected = detect_available_frameworks(working_directory)

    if detected:
        # Return highest confidence detection
        best = max(detected, key=lambda d: d.confidence)
        LOG.info("Recommending %s based on existing setup", best.framework.value)
        return best

    # Fall back to language-based recommendation
    lang = detect_language(working_directory)

    if lang.primary == "python":
        return FrameworkDetection(
            framework=MutationFramework.MUTMUT,
            confidence=0.6,
            evidence=f"Python project detected (no mutation tool configured yet)",
        )

    if lang.primary in ("javascript", "typescript"):
        return FrameworkDetection(
            framework=MutationFramework.STRYKER,
            confidence=0.6,
            evidence=f"{lang.primary.title()} project detected (no mutation tool configured yet)",
        )

    if lang.primary == "java":
        return FrameworkDetection(
            framework=MutationFramework.UNKNOWN,
            confidence=0.3,
            evidence="Java project - consider PIT mutation testing (not yet supported)",
        )

    return FrameworkDetection(
        framework=MutationFramework.UNKNOWN,
        confidence=0.0,
        evidence=f"Could not determine appropriate framework for {lang.primary} project",
    )
