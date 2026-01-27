"""
Hierarchical code region model.

Inspired by pytest's node ID format (well-established, familiar to Python developers).
Supports serialization to/from strings for storage compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


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

    Format:
        file:<path>                              # Whole file
        file:<path>::class:<name>                # Class level
        file:<path>::func:<name>                 # Function level
        file:<path>::class:<name>::func:<name>   # Method within class
        file:<path>::lines:<start>-<end>         # Line range

    Examples:
        >>> Region.for_file("src/main.py")
        Region('file:src/main.py')

        >>> Region.for_function("src/main.py", "compute", class_name="Calculator")
        Region('file:src/main.py::class:Calculator::func:compute')

        >>> r = Region.from_string("file:src/main.py::lines:10-20")
        >>> r.line_start
        10
    """

    file_path: str
    class_name: Optional[str] = None
    func_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate region fields."""
        if not self.file_path:
            raise ValueError("file_path cannot be empty")

        if self.line_start is not None and self.line_end is not None:
            if self.line_start > self.line_end:
                raise ValueError(
                    f"line_start ({self.line_start}) cannot be greater than "
                    f"line_end ({self.line_end})"
                )
            if self.line_start < 1:
                raise ValueError(f"line_start must be >= 1, got {self.line_start}")

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
        """
        Check if this region contains another region.

        A region contains another if:
        - Same file AND
        - This region's scope encompasses the other's scope

        Examples:
            file:a.py contains file:a.py::func:foo  # True
            file:a.py::class:C contains file:a.py::class:C::func:m  # True
            file:a.py::func:foo contains file:a.py  # False
        """
        # Must be same file
        if self.file_path != other.file_path:
            return False

        # File contains everything in that file
        if self.level == RegionLevel.FILE:
            return True

        # Class level containment
        if self.level == RegionLevel.CLASS:
            # Other must be in same class (or be the class itself)
            if other.class_name != self.class_name:
                return False
            # Class contains its methods and line ranges within it
            return True

        # Function level containment
        if self.level == RegionLevel.FUNCTION:
            # Must match both class and function
            if other.func_name != self.func_name:
                return False
            if other.class_name != self.class_name:
                return False
            # Function contains line ranges within it
            return True

        # Lines level containment
        if self.level == RegionLevel.LINES and other.level == RegionLevel.LINES:
            if self.line_start is None or self.line_end is None:
                return False
            if other.line_start is None or other.line_end is None:
                return False
            return (
                self.line_start <= other.line_start
                and self.line_end >= other.line_end
            )

        return False

    def overlaps(self, other: "Region") -> bool:
        """
        Check if this region overlaps with another.

        Two regions overlap if they share any code locations.
        """
        if self.file_path != other.file_path:
            return False

        # If either contains the other, they overlap
        if self.contains(other) or other.contains(self):
            return True

        # Line range overlap check
        if (
            self.line_start is not None
            and self.line_end is not None
            and other.line_start is not None
            and other.line_end is not None
        ):
            # Check for non-overlap and negate
            no_overlap = (
                self.line_end < other.line_start or other.line_end < self.line_start
            )
            return not no_overlap

        # Same function (regardless of line ranges)
        if self.func_name and self.func_name == other.func_name:
            if self.class_name == other.class_name:
                return True

        # Same class (regardless of function)
        if self.class_name and self.class_name == other.class_name:
            # Only if neither specifies a different function
            if self.func_name is None or other.func_name is None:
                return True
            if self.func_name == other.func_name:
                return True

        return False

    def to_string(self) -> str:
        """
        Serialize to canonical string format.

        The format is designed to be:
        - Human readable
        - Lexicographically sortable by file
        - Parseable back to Region
        """
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
        """
        Parse from canonical string format.

        Also handles legacy plain file paths for backward compatibility.

        Args:
            s: Region string like "file:src/main.py::func:compute"
               or legacy plain path like "src/main.py"

        Returns:
            Parsed Region instance

        Raises:
            ValueError: If string is empty or malformed
        """
        if not s:
            raise ValueError("Empty region string")

        s = s.strip()

        # Handle legacy plain strings (backward compatibility)
        if not s.startswith("file:"):
            # Assume it's a file path or legacy region ID
            return cls(file_path=s)

        file_path: Optional[str] = None
        class_name: Optional[str] = None
        func_name: Optional[str] = None
        line_start: Optional[int] = None
        line_end: Optional[int] = None

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
    def for_class(cls, path: str, class_name: str) -> "Region":
        """Create a class-level region."""
        return cls(file_path=path, class_name=class_name)

    @classmethod
    def for_function(
        cls,
        path: str,
        func_name: str,
        class_name: Optional[str] = None,
    ) -> "Region":
        """Create a function-level region."""
        return cls(file_path=path, class_name=class_name, func_name=func_name)

    @classmethod
    def for_lines(
        cls,
        path: str,
        start: int,
        end: int,
        func_name: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> "Region":
        """Create a line-range region."""
        return cls(
            file_path=path,
            class_name=class_name,
            func_name=func_name,
            line_start=start,
            line_end=end,
        )

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"Region({self.to_string()!r})"


def normalize_region_id(region_id: Optional[str]) -> Optional[Region]:
    """
    Convert legacy regionId string to Region, or return None.

    Provides backward compatibility with existing regionId: str fields.
    """
    if region_id is None:
        return None
    return Region.from_string(region_id)


def region_to_id(region: Optional[Region]) -> Optional[str]:
    """
    Convert Region back to string for storage.

    Inverse of normalize_region_id.
    """
    if region is None:
        return None
    return region.to_string()
