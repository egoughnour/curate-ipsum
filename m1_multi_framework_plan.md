# M1: Multi-Framework Foundation - Implementation Plan

## Objective

Add flexible region definitions and multi-framework mutation tool support without breaking existing functionality or introducing throwaway code.

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| regionId | `Optional[str]` | Unstructured, no semantic meaning |
| Stryker parser | ✅ Working | JavaScript mutation tool |
| mutmut parser | ❌ Missing | Most popular Python mutation tool |
| Framework detection | ❌ Missing | No auto-detection |
| Region hierarchy | ❌ Missing | No file → function → line structure |

## Design Principles

1. **No placeholders** - Every function computes real values
2. **Backward compatible** - Existing `regionId: str` continues to work
3. **Framework agnostic** - Region model works across all tools
4. **Incrementally useful** - Each piece delivers value independently

---

## Part 1: Flexible Region Model

### Region Addressing Scheme

Inspired by pytest's node ID format (well-established, familiar to Python developers):

```
file:<path>                           # Whole file
file:<path>::class:<name>             # Class level
file:<path>::func:<name>              # Function/method level
file:<path>::lines:<start>-<end>      # Line range
file:<path>::func:<name>::lines:<s>-<e>  # Lines within function
```

### Examples

```python
# Whole file
"file:src/calculator.py"

# Specific function
"file:src/calculator.py::func:compute_total"

# Class
"file:src/calculator.py::class:Calculator"

# Method within class
"file:src/calculator.py::class:Calculator::func:add"

# Line range (for fine-grained mutation tracking)
"file:src/calculator.py::lines:45-52"

# Lines within a function
"file:src/calculator.py::func:compute_total::lines:10-15"
```

### Region Model

```python
# regions/models.py

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import re


class RegionLevel(str, Enum):
    """Granularity level of a region."""
    FILE = "file"
    CLASS = "class"
    FUNCTION = "func"
    LINES = "lines"


@dataclass(frozen=True)
class Region:
    """
    Hierarchical code region identifier.

    Immutable, hashable, and serializable to/from string.
    Supports containment queries (is region A within region B?).
    """
    file_path: str
    class_name: Optional[str] = None
    func_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    @property
    def level(self) -> RegionLevel:
        """Most specific level of this region."""
        if self.line_start is not None:
            return RegionLevel.LINES
        if self.func_name is not None:
            return RegionLevel.FUNCTION
        if self.class_name is not None:
            return RegionLevel.CLASS
        return RegionLevel.FILE

    def contains(self, other: "Region") -> bool:
        """Check if this region contains another region."""
        # Must be same file
        if self.file_path != other.file_path:
            return False

        # File contains everything in that file
        if self.level == RegionLevel.FILE:
            return True

        # Class contains functions/lines within it
        if self.level == RegionLevel.CLASS:
            if other.class_name != self.class_name:
                return False
            return True

        # Function contains lines within it
        if self.level == RegionLevel.FUNCTION:
            if other.func_name != self.func_name:
                return False
            if other.class_name != self.class_name:
                return False
            return True

        # Lines containment
        if self.level == RegionLevel.LINES and other.level == RegionLevel.LINES:
            if self.line_start is None or self.line_end is None:
                return False
            if other.line_start is None or other.line_end is None:
                return False
            return (self.line_start <= other.line_start and
                    self.line_end >= other.line_end)

        return False

    def overlaps(self, other: "Region") -> bool:
        """Check if this region overlaps with another."""
        if self.file_path != other.file_path:
            return False

        # If either contains the other, they overlap
        if self.contains(other) or other.contains(self):
            return True

        # Line range overlap
        if (self.line_start is not None and self.line_end is not None and
            other.line_start is not None and other.line_end is not None):
            return not (self.line_end < other.line_start or
                       other.line_end < self.line_start)

        # Same function or class
        if self.func_name and self.func_name == other.func_name:
            return True
        if self.class_name and self.class_name == other.class_name:
            return True

        return False

    def to_string(self) -> str:
        """Serialize to canonical string format."""
        parts = [f"file:{self.file_path}"]

        if self.class_name:
            parts.append(f"class:{self.class_name}")

        if self.func_name:
            parts.append(f"func:{self.func_name}")

        if self.line_start is not None:
            if self.line_end is not None and self.line_end != self.line_start:
                parts.append(f"lines:{self.line_start}-{self.line_end}")
            else:
                parts.append(f"lines:{self.line_start}-{self.line_start}")

        return "::".join(parts)

    @classmethod
    def from_string(cls, s: str) -> "Region":
        """Parse from canonical string format."""
        if not s:
            raise ValueError("Empty region string")

        # Handle legacy plain strings (backward compatibility)
        if not s.startswith("file:"):
            # Assume it's a file path or legacy region ID
            return cls(file_path=s)

        file_path = None
        class_name = None
        func_name = None
        line_start = None
        line_end = None

        parts = s.split("::")
        for part in parts:
            if part.startswith("file:"):
                file_path = part[5:]
            elif part.startswith("class:"):
                class_name = part[6:]
            elif part.startswith("func:"):
                func_name = part[5:]
            elif part.startswith("lines:"):
                line_spec = part[6:]
                if "-" in line_spec:
                    start_s, end_s = line_spec.split("-", 1)
                    line_start = int(start_s)
                    line_end = int(end_s)
                else:
                    line_start = line_end = int(line_spec)

        if file_path is None:
            raise ValueError(f"No file path in region string: {s}")

        return cls(
            file_path=file_path,
            class_name=class_name,
            func_name=func_name,
            line_start=line_start,
            line_end=line_end,
        )

    @classmethod
    def for_file(cls, path: str) -> "Region":
        """Create a file-level region."""
        return cls(file_path=path)

    @classmethod
    def for_function(cls, path: str, func_name: str,
                     class_name: Optional[str] = None) -> "Region":
        """Create a function-level region."""
        return cls(file_path=path, class_name=class_name, func_name=func_name)

    @classmethod
    def for_lines(cls, path: str, start: int, end: int,
                  func_name: Optional[str] = None,
                  class_name: Optional[str] = None) -> "Region":
        """Create a line-range region."""
        return cls(
            file_path=path,
            class_name=class_name,
            func_name=func_name,
            line_start=start,
            line_end=end
        )

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"Region({self.to_string()!r})"
```

### Backward Compatibility

The existing `regionId: Optional[str]` field remains. The Region model provides:

```python
def normalize_region_id(region_id: Optional[str]) -> Optional[Region]:
    """Convert legacy regionId to Region, or return None."""
    if region_id is None:
        return None
    return Region.from_string(region_id)

def region_to_id(region: Optional[Region]) -> Optional[str]:
    """Convert Region back to string for storage."""
    if region is None:
        return None
    return region.to_string()
```

---

## Part 2: Mutmut Parser

### Mutmut Data Model (from research)

Mutmut uses SQLite cache (`.mutmut-cache`) with:

| Table | Fields |
|-------|--------|
| `SourceFile` | filename, hash |
| `Line` | sourcefile_id, line, line_number |
| `Mutant` | line_id, index, tested_against_hash, status |

**Status values:**
- `OK_KILLED` - Mutant killed by tests
- `BAD_SURVIVED` - Mutant survived
- `BAD_TIMEOUT` - Test timed out
- `OK_SUSPICIOUS` - Suspicious result
- `UNTESTED` - Not yet tested

### Parser Implementation

```python
# parsers/mutmut_parser.py

from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models import FileMutationStats
from regions.models import Region


# Mutmut status value mapping
class MutmutStatus:
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
    index: int  # Mutant index within line


def find_mutmut_cache(working_directory: str) -> Optional[Path]:
    """
    Locate the mutmut cache file.

    Searches in order:
    1. .mutmut-cache in working directory
    2. .mutmut-cache in parent directories (up to 3 levels)
    """
    cwd = Path(working_directory)

    for parent in [cwd] + list(cwd.parents)[:3]:
        cache_path = parent / ".mutmut-cache"
        if cache_path.exists():
            return cache_path

    return None


def parse_mutmut_cache(cache_path: Path) -> List[MutmutMutant]:
    """
    Parse mutmut SQLite cache and extract all mutants.

    The cache schema:
    - MutantEntry table (or similar) contains mutant data
    - Links to source files and line numbers
    """
    mutants: List[MutmutMutant] = []

    conn = sqlite3.connect(str(cache_path))
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # Query the mutmut schema - this may vary by version
        # Try the known schema first
        try:
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
        except sqlite3.OperationalError:
            # Try alternative schema (older mutmut versions)
            cursor.execute("""
                SELECT
                    m.id,
                    m.filename,
                    m.line_number,
                    m.status,
                    m.mutation_index as "index"
                FROM mutant m
            """)

        for row in cursor.fetchall():
            mutants.append(MutmutMutant(
                id=row["id"],
                file_path=row["filename"],
                line_number=row["line_number"],
                status=row["status"].lower() if row["status"] else MutmutStatus.UNTESTED,
                index=row["index"] or 0,
            ))

    finally:
        conn.close()

    return mutants


def aggregate_mutmut_stats(
    mutants: List[MutmutMutant]
) -> Tuple[int, int, int, int, float, List[FileMutationStats]]:
    """
    Aggregate mutant list into summary statistics.

    Returns: (total, killed, survived, no_coverage, score, by_file)
    """
    # Group by file
    by_file: Dict[str, List[MutmutMutant]] = {}
    for m in mutants:
        by_file.setdefault(m.file_path, []).append(m)

    file_stats: List[FileMutationStats] = []
    total_killed = 0
    total_survived = 0
    total_timeout = 0  # Treated as "no coverage" equivalent
    total_untested = 0

    for file_path, file_mutants in sorted(by_file.items()):
        killed = sum(1 for m in file_mutants if m.status == MutmutStatus.OK_KILLED)
        survived = sum(1 for m in file_mutants
                      if m.status in (MutmutStatus.BAD_SURVIVED, MutmutStatus.OK_SUSPICIOUS))
        timeout = sum(1 for m in file_mutants if m.status == MutmutStatus.BAD_TIMEOUT)
        untested = sum(1 for m in file_mutants
                      if m.status in (MutmutStatus.UNTESTED, MutmutStatus.SKIPPED))

        total = len(file_mutants)

        # Mutation score: killed / (killed + survived)
        # Timeout and untested are excluded from score calculation
        denominator = killed + survived
        score = killed / denominator if denominator > 0 else 0.0

        file_stats.append(FileMutationStats(
            filePath=file_path,
            totalMutants=total,
            killed=killed,
            survived=survived,
            noCoverage=timeout + untested,  # Combine into noCoverage
            mutationScore=score,
        ))

        total_killed += killed
        total_survived += survived
        total_timeout += timeout
        total_untested += untested

    total = len(mutants)
    denominator = total_killed + total_survived
    overall_score = total_killed / denominator if denominator > 0 else 0.0

    return (
        total,
        total_killed,
        total_survived,
        total_timeout + total_untested,  # noCoverage
        overall_score,
        file_stats,
    )


def parse_mutmut_output(
    working_directory: str,
    cache_path: Optional[str] = None,
) -> Tuple[int, int, int, int, float, List[FileMutationStats]]:
    """
    Parse mutmut results from cache.

    This is the main entry point, matching the signature of parse_stryker_output.
    """
    if cache_path:
        cache = Path(cache_path)
    else:
        cache = find_mutmut_cache(working_directory)

    if cache is None or not cache.exists():
        raise FileNotFoundError(
            f"Mutmut cache not found. Run 'mutmut run' first. "
            f"Searched in: {working_directory}"
        )

    mutants = parse_mutmut_cache(cache)

    if not mutants:
        # Empty cache - return zeros
        return (0, 0, 0, 0, 0.0, [])

    return aggregate_mutmut_stats(mutants)


def get_mutmut_region_mutants(
    working_directory: str,
    region: Region,
    cache_path: Optional[str] = None,
) -> List[MutmutMutant]:
    """
    Get mutants within a specific region.

    Useful for region-level mutation score calculation.
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
        mutant_region = Region.for_lines(m.file_path, m.line_number, m.line_number)
        if region.contains(mutant_region) or region.overlaps(mutant_region):
            result.append(m)

    return result
```

---

## Part 3: Framework Auto-Detection

### Detection Strategy

```python
# parsers/detection.py

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set


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


def detect_language(working_directory: str) -> ProjectLanguage:
    """
    Detect primary language of a project.

    Examines file extensions and configuration files.
    """
    cwd = Path(working_directory)

    # Count files by extension
    ext_counts: dict[str, int] = {}
    for f in cwd.rglob("*"):
        if f.is_file() and not any(p.startswith(".") for p in f.parts):
            ext = f.suffix.lower()
            if ext:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # Language mapping
    lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".rs": "rust",
        ".go": "go",
        ".rb": "ruby",
        ".cs": "csharp",
    }

    # Config file signals
    config_signals = {
        "pyproject.toml": "python",
        "setup.py": "python",
        "requirements.txt": "python",
        "package.json": "javascript",
        "tsconfig.json": "typescript",
        "Cargo.toml": "rust",
        "go.mod": "go",
        "pom.xml": "java",
        "build.gradle": "java",
        "Gemfile": "ruby",
    }

    # Score languages
    lang_scores: dict[str, float] = {}

    for ext, count in ext_counts.items():
        lang = lang_map.get(ext)
        if lang:
            lang_scores[lang] = lang_scores.get(lang, 0) + count

    # Boost from config files
    for config, lang in config_signals.items():
        if (cwd / config).exists():
            lang_scores[lang] = lang_scores.get(lang, 0) + 100  # Strong signal

    if not lang_scores:
        return ProjectLanguage(primary="unknown", secondary=[], confidence=0.0)

    # Sort by score
    sorted_langs = sorted(lang_scores.items(), key=lambda x: -x[1])
    total = sum(lang_scores.values())

    primary = sorted_langs[0][0]
    confidence = sorted_langs[0][1] / total if total > 0 else 0.0
    secondary = [lang for lang, _ in sorted_langs[1:4]]

    return ProjectLanguage(
        primary=primary,
        secondary=secondary,
        confidence=min(1.0, confidence),
    )


def detect_available_frameworks(working_directory: str) -> List[FrameworkDetection]:
    """
    Detect which mutation frameworks have been run or are configured.

    Checks for:
    - Output files/directories from each framework
    - Configuration files
    - Cache files
    """
    cwd = Path(working_directory)
    detections: List[FrameworkDetection] = []

    # Stryker detection
    stryker_signals = [
        cwd / "stryker.conf.js",
        cwd / "stryker.conf.json",
        cwd / ".stryker-tmp",
        cwd / "reports" / "mutation" / "mutation.json",
        cwd / "reports" / "stryker-report.json",
    ]
    for signal in stryker_signals:
        if signal.exists():
            detections.append(FrameworkDetection(
                framework=MutationFramework.STRYKER,
                confidence=0.9 if "report" in str(signal) else 0.7,
                evidence=f"Found {signal.name}",
            ))
            break

    # Mutmut detection
    mutmut_signals = [
        cwd / ".mutmut-cache",
        cwd / "mutmut.toml",
        cwd / "setup.cfg",  # May contain [mutmut] section
    ]
    for signal in mutmut_signals:
        if signal.exists():
            if signal.name == "setup.cfg":
                # Check for [mutmut] section
                content = signal.read_text()
                if "[mutmut]" in content:
                    detections.append(FrameworkDetection(
                        framework=MutationFramework.MUTMUT,
                        confidence=0.8,
                        evidence="Found [mutmut] in setup.cfg",
                    ))
                    break
            else:
                detections.append(FrameworkDetection(
                    framework=MutationFramework.MUTMUT,
                    confidence=0.9 if signal.name == ".mutmut-cache" else 0.7,
                    evidence=f"Found {signal.name}",
                ))
                break

    # Cosmic-ray detection
    cosmic_signals = [
        cwd / ".cosmic-ray.toml",
        cwd / "cosmic-ray.toml",
    ]
    for signal in cosmic_signals:
        if signal.exists():
            detections.append(FrameworkDetection(
                framework=MutationFramework.COSMIC_RAY,
                confidence=0.8,
                evidence=f"Found {signal.name}",
            ))
            break

    return detections


def recommend_framework(working_directory: str) -> FrameworkDetection:
    """
    Recommend the best mutation framework for a project.

    Based on:
    1. Already-run frameworks (highest priority)
    2. Project language
    3. Available configuration
    """
    # Check what's already been run
    detected = detect_available_frameworks(working_directory)
    if detected:
        # Return highest confidence detection
        return max(detected, key=lambda d: d.confidence)

    # Fall back to language-based recommendation
    lang = detect_language(working_directory)

    if lang.primary == "python":
        return FrameworkDetection(
            framework=MutationFramework.MUTMUT,
            confidence=0.6,
            evidence=f"Python project detected (no mutation cache found)",
        )
    elif lang.primary in ("javascript", "typescript"):
        return FrameworkDetection(
            framework=MutationFramework.STRYKER,
            confidence=0.6,
            evidence=f"JavaScript/TypeScript project detected (no mutation cache found)",
        )

    return FrameworkDetection(
        framework=MutationFramework.UNKNOWN,
        confidence=0.0,
        evidence="Could not determine appropriate framework",
    )
```

---

## Part 4: Unified Parser Interface

### Router Implementation

```python
# parsers/__init__.py

from __future__ import annotations
from typing import List, Optional, Tuple

from models import FileMutationStats
from parsers.detection import (
    MutationFramework,
    detect_available_frameworks,
    recommend_framework,
)
from parsers.mutmut_parser import parse_mutmut_output
from parsers.stryker_parser import parse_stryker_output  # Extracted from tools.py


class UnsupportedFrameworkError(Exception):
    """Raised when a mutation framework is not supported."""
    pass


def parse_mutation_output(
    working_directory: str,
    tool: Optional[str] = None,
    report_path: Optional[str] = None,
) -> Tuple[int, int, int, int, float, List[FileMutationStats]]:
    """
    Parse mutation testing output, auto-detecting framework if not specified.

    Args:
        working_directory: Project directory
        tool: Optional framework name (auto-detected if None)
        report_path: Optional path to report file/cache

    Returns:
        Tuple of (total, killed, survived, no_coverage, score, by_file)

    Raises:
        UnsupportedFrameworkError: If framework is not supported
        FileNotFoundError: If report/cache not found
    """
    # Auto-detect if not specified
    if tool is None:
        detection = recommend_framework(working_directory)
        tool = detection.framework.value

    tool_lower = tool.lower()

    if tool_lower == MutationFramework.STRYKER.value:
        return parse_stryker_output(report_path, working_directory)

    elif tool_lower == MutationFramework.MUTMUT.value:
        return parse_mutmut_output(working_directory, report_path)

    elif tool_lower == MutationFramework.COSMIC_RAY.value:
        # TODO: Implement cosmic-ray parser
        raise UnsupportedFrameworkError(
            f"cosmic-ray parser not yet implemented. "
            f"Supported: stryker, mutmut"
        )

    else:
        raise UnsupportedFrameworkError(
            f"Unknown mutation framework: {tool}. "
            f"Supported: stryker, mutmut"
        )
```

---

## Part 5: MCP Tool Updates

### New/Modified Tools

```python
# In server.py

@server.tool(
    description=(
        "Detect available mutation testing frameworks and project language. "
        "Returns recommendations for which framework to use."
    )
)
def detect_frameworks_tool(workingDirectory: str) -> dict:
    """Detect mutation frameworks in a project."""
    from parsers.detection import (
        detect_available_frameworks,
        detect_language,
        recommend_framework,
    )

    language = detect_language(workingDirectory)
    frameworks = detect_available_frameworks(workingDirectory)
    recommendation = recommend_framework(workingDirectory)

    return {
        "language": {
            "primary": language.primary,
            "secondary": language.secondary,
            "confidence": language.confidence,
        },
        "detected_frameworks": [
            {
                "framework": f.framework.value,
                "confidence": f.confidence,
                "evidence": f.evidence,
            }
            for f in frameworks
        ],
        "recommendation": {
            "framework": recommendation.framework.value,
            "confidence": recommendation.confidence,
            "evidence": recommendation.evidence,
        },
    }


@server.tool(
    description=(
        "Parse a region identifier string into its components. "
        "Useful for understanding region hierarchy and containment."
    )
)
def parse_region_tool(regionId: str) -> dict:
    """Parse a region string into components."""
    from regions.models import Region

    region = Region.from_string(regionId)

    return {
        "regionId": region.to_string(),
        "level": region.level.value,
        "file_path": region.file_path,
        "class_name": region.class_name,
        "func_name": region.func_name,
        "line_start": region.line_start,
        "line_end": region.line_end,
    }


@server.tool(
    description=(
        "Check if one region contains or overlaps another. "
        "Useful for aggregating metrics across related regions."
    )
)
def check_region_relationship_tool(
    regionA: str,
    regionB: str,
) -> dict:
    """Check containment/overlap relationship between regions."""
    from regions.models import Region

    a = Region.from_string(regionA)
    b = Region.from_string(regionB)

    return {
        "a": a.to_string(),
        "b": b.to_string(),
        "a_contains_b": a.contains(b),
        "b_contains_a": b.contains(a),
        "overlaps": a.overlaps(b),
    }
```

### Modified run_mutation_tests

```python
# Modified signature in tools.py

async def run_mutation_tests(
    projectId: str,
    commitSha: str,
    command: str,
    workingDirectory: str,
    regionId: Optional[str] = None,
    tool: Optional[str] = None,  # Changed: now optional, auto-detected
    reportPath: Optional[str] = None,
) -> MutationRunResult:
    """
    Run mutation tests and parse results.

    If tool is not specified, auto-detects based on project structure.
    """
    result = await run_command(command, workingDirectory)

    # Use unified parser with auto-detection
    from parsers import parse_mutation_output

    total_mutants, killed, survived, no_coverage, mutation_score, by_file = \
        parse_mutation_output(
            working_directory=workingDirectory,
            tool=tool,
            report_path=reportPath,
        )

    # Detect actual tool used if not specified
    if tool is None:
        from parsers.detection import recommend_framework
        tool = recommend_framework(workingDirectory).framework.value

    # ... rest of function unchanged
```

---

## File Structure

```
curate-ipsum/
├── regions/
│   ├── __init__.py
│   └── models.py          # Region, RegionLevel
├── parsers/
│   ├── __init__.py        # Unified interface, parse_mutation_output
│   ├── detection.py       # Framework/language detection
│   ├── stryker_parser.py  # Extracted from tools.py
│   └── mutmut_parser.py   # New mutmut parser
├── models.py              # Unchanged (regionId remains str)
├── tools.py               # Modified to use unified parser
└── server.py              # New MCP tools added
```

---

## Implementation Order

| Phase | Task | Dependencies | LOC |
|-------|------|--------------|-----|
| 1 | Region model | None | ~150 |
| 2 | Extract stryker_parser.py | None | ~80 |
| 3 | Framework detection | None | ~120 |
| 4 | Mutmut parser | Region model | ~150 |
| 5 | Unified parser interface | All parsers | ~50 |
| 6 | MCP tool updates | All above | ~80 |
| 7 | Tests | All above | ~200 |

**Total: ~830 lines of production code**

---

## Testing Strategy

```python
# tests/test_regions.py

def test_region_parsing_roundtrip():
    """Regions serialize and deserialize correctly."""
    cases = [
        "file:src/main.py",
        "file:src/main.py::class:Calculator",
        "file:src/main.py::func:compute",
        "file:src/main.py::class:Calculator::func:add",
        "file:src/main.py::lines:10-25",
    ]
    for case in cases:
        region = Region.from_string(case)
        assert region.to_string() == case


def test_region_containment():
    """Containment relationships are correct."""
    file_region = Region.for_file("src/main.py")
    func_region = Region.for_function("src/main.py", "compute")
    line_region = Region.for_lines("src/main.py", 10, 20)

    assert file_region.contains(func_region)
    assert file_region.contains(line_region)
    assert not func_region.contains(file_region)


def test_backward_compatibility():
    """Legacy plain string regionIds still work."""
    region = Region.from_string("src/some/path.py")
    assert region.file_path == "src/some/path.py"
    assert region.level == RegionLevel.FILE
```

---

## Success Criteria

1. ✅ Region model supports file/class/function/lines levels
2. ✅ Regions are serializable to/from strings
3. ✅ Containment and overlap queries work correctly
4. ✅ Backward compatible with existing `regionId: str`
5. ✅ Mutmut cache parsing works
6. ✅ Framework auto-detection returns sensible results
7. ✅ Unified parser routes to correct implementation
8. ✅ All tests pass
9. ✅ No placeholders or hardcoded constants

---

## Sources

- [mutmut GitHub](https://github.com/boxed/mutmut)
- [mutmut cache schema (Snyk)](https://snyk.io/advisor/python/mutmut/functions/mutmut.cache.Mutant)
- [Stryker documentation](https://stryker-mutator.io/)
