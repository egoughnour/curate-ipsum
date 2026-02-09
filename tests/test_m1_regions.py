"""
Tests for the Region model (M1: Multi-Framework Foundation).

Tests cover:
- Region string parsing and serialization
- Containment relationships
- Overlap detection
- Factory methods
- Backward compatibility with legacy regionId strings
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from regions.models import Region, RegionLevel


class TestRegionParsing:
    """Tests for Region.from_string() and to_string()."""

    def test_file_level_roundtrip(self):
        """File-level regions serialize and deserialize correctly."""
        original = "file:src/main.py"
        region = Region.from_string(original)

        assert region.file_path == "src/main.py"
        assert region.class_name is None
        assert region.func_name is None
        assert region.line_start is None
        assert region.level == RegionLevel.FILE
        assert region.to_string() == original

    def test_class_level_roundtrip(self):
        """Class-level regions serialize and deserialize correctly."""
        original = "file:src/main.py::class:Calculator"
        region = Region.from_string(original)

        assert region.file_path == "src/main.py"
        assert region.class_name == "Calculator"
        assert region.func_name is None
        assert region.level == RegionLevel.CLASS
        assert region.to_string() == original

    def test_function_level_roundtrip(self):
        """Function-level regions serialize and deserialize correctly."""
        original = "file:src/main.py::func:compute"
        region = Region.from_string(original)

        assert region.file_path == "src/main.py"
        assert region.func_name == "compute"
        assert region.level == RegionLevel.FUNCTION
        assert region.to_string() == original

    def test_method_level_roundtrip(self):
        """Method-level regions (class + func) serialize correctly."""
        original = "file:src/main.py::class:Calculator::func:add"
        region = Region.from_string(original)

        assert region.file_path == "src/main.py"
        assert region.class_name == "Calculator"
        assert region.func_name == "add"
        assert region.level == RegionLevel.FUNCTION
        assert region.to_string() == original

    def test_lines_level_roundtrip(self):
        """Line-range regions serialize and deserialize correctly."""
        original = "file:src/main.py::lines:10-25"
        region = Region.from_string(original)

        assert region.file_path == "src/main.py"
        assert region.line_start == 10
        assert region.line_end == 25
        assert region.level == RegionLevel.LINES
        assert region.to_string() == original

    def test_single_line_roundtrip(self):
        """Single-line regions normalize to lines:N-N format."""
        region = Region.for_lines("src/main.py", 15, 15)

        assert region.line_start == 15
        assert region.line_end == 15
        assert region.to_string() == "file:src/main.py::lines:15-15"

    def test_complex_region_roundtrip(self):
        """Complex regions with all components serialize correctly."""
        original = "file:src/main.py::class:Calculator::func:add::lines:10-20"
        region = Region.from_string(original)

        assert region.file_path == "src/main.py"
        assert region.class_name == "Calculator"
        assert region.func_name == "add"
        assert region.line_start == 10
        assert region.line_end == 20
        assert region.to_string() == original

    def test_backward_compatibility_plain_path(self):
        """Legacy plain file paths are treated as file-level regions."""
        region = Region.from_string("src/some/path.py")

        assert region.file_path == "src/some/path.py"
        assert region.level == RegionLevel.FILE
        # Note: to_string() will add the file: prefix
        assert "src/some/path.py" in region.to_string()

    def test_backward_compatibility_legacy_region_id(self):
        """Legacy region IDs without file: prefix are treated as file paths."""
        region = Region.from_string("my_module::my_function")

        # Treated as a file path (backward compat)
        assert region.file_path == "my_module::my_function"

    def test_empty_string_raises(self):
        """Empty strings raise ValueError."""
        with pytest.raises(ValueError, match="Empty region string"):
            Region.from_string("")

    def test_whitespace_only_raises(self):
        """Whitespace-only strings raise ValueError."""
        with pytest.raises(ValueError, match="Empty region string"):
            Region.from_string("   ")


class TestRegionContainment:
    """Tests for Region.contains()."""

    def test_file_contains_function(self):
        """File regions contain functions within them."""
        file_region = Region.for_file("src/main.py")
        func_region = Region.for_function("src/main.py", "compute")

        assert file_region.contains(func_region)
        assert not func_region.contains(file_region)

    def test_file_contains_class(self):
        """File regions contain classes within them."""
        file_region = Region.for_file("src/main.py")
        class_region = Region.for_class("src/main.py", "Calculator")

        assert file_region.contains(class_region)
        assert not class_region.contains(file_region)

    def test_file_contains_lines(self):
        """File regions contain line ranges within them."""
        file_region = Region.for_file("src/main.py")
        lines_region = Region.for_lines("src/main.py", 10, 20)

        assert file_region.contains(lines_region)
        assert not lines_region.contains(file_region)

    def test_class_contains_method(self):
        """Class regions contain methods within them."""
        class_region = Region.for_class("src/main.py", "Calculator")
        method_region = Region.for_function("src/main.py", "add", class_name="Calculator")

        assert class_region.contains(method_region)
        assert not method_region.contains(class_region)

    def test_class_does_not_contain_different_class_method(self):
        """Class regions don't contain methods from different classes."""
        class_region = Region.for_class("src/main.py", "Calculator")
        other_method = Region.for_function("src/main.py", "helper", class_name="Utils")

        assert not class_region.contains(other_method)

    def test_function_contains_lines_within(self):
        """Function regions contain line ranges within them."""
        func_region = Region.for_function("src/main.py", "compute")
        # Note: This is a semantic containment - the function "owns" its lines
        lines_region = Region.for_lines("src/main.py", 10, 20, func_name="compute")

        assert func_region.contains(lines_region)

    def test_lines_contain_subset(self):
        """Line ranges contain subsets of themselves."""
        outer = Region.for_lines("src/main.py", 10, 30)
        inner = Region.for_lines("src/main.py", 15, 25)

        assert outer.contains(inner)
        assert not inner.contains(outer)

    def test_different_files_no_containment(self):
        """Regions in different files don't contain each other."""
        region_a = Region.for_file("src/a.py")
        region_b = Region.for_function("src/b.py", "func")

        assert not region_a.contains(region_b)
        assert not region_b.contains(region_a)

    def test_self_containment(self):
        """A region contains itself."""
        region = Region.for_function("src/main.py", "compute")

        assert region.contains(region)


class TestRegionOverlap:
    """Tests for Region.overlaps()."""

    def test_containment_implies_overlap(self):
        """If A contains B, they overlap."""
        file_region = Region.for_file("src/main.py")
        func_region = Region.for_function("src/main.py", "compute")

        assert file_region.overlaps(func_region)
        assert func_region.overlaps(file_region)

    def test_line_ranges_overlap(self):
        """Overlapping line ranges are detected."""
        region_a = Region.for_lines("src/main.py", 10, 20)
        region_b = Region.for_lines("src/main.py", 15, 25)

        assert region_a.overlaps(region_b)
        assert region_b.overlaps(region_a)

    def test_line_ranges_no_overlap(self):
        """Non-overlapping line ranges are detected."""
        region_a = Region.for_lines("src/main.py", 10, 20)
        region_b = Region.for_lines("src/main.py", 25, 35)

        assert not region_a.overlaps(region_b)
        assert not region_b.overlaps(region_a)

    def test_adjacent_lines_no_overlap(self):
        """Adjacent but non-overlapping line ranges don't overlap."""
        region_a = Region.for_lines("src/main.py", 10, 20)
        region_b = Region.for_lines("src/main.py", 21, 30)

        assert not region_a.overlaps(region_b)

    def test_same_function_overlaps(self):
        """Same function in same class overlaps."""
        region_a = Region.for_function("src/main.py", "compute", class_name="Calc")
        region_b = Region.for_function("src/main.py", "compute", class_name="Calc")

        assert region_a.overlaps(region_b)

    def test_different_files_no_overlap(self):
        """Regions in different files don't overlap."""
        region_a = Region.for_file("src/a.py")
        region_b = Region.for_file("src/b.py")

        assert not region_a.overlaps(region_b)


class TestRegionFactories:
    """Tests for Region factory methods."""

    def test_for_file(self):
        """Region.for_file() creates file-level regions."""
        region = Region.for_file("src/main.py")

        assert region.file_path == "src/main.py"
        assert region.level == RegionLevel.FILE

    def test_for_class(self):
        """Region.for_class() creates class-level regions."""
        region = Region.for_class("src/main.py", "Calculator")

        assert region.file_path == "src/main.py"
        assert region.class_name == "Calculator"
        assert region.level == RegionLevel.CLASS

    def test_for_function_standalone(self):
        """Region.for_function() creates function-level regions."""
        region = Region.for_function("src/main.py", "compute")

        assert region.file_path == "src/main.py"
        assert region.func_name == "compute"
        assert region.class_name is None
        assert region.level == RegionLevel.FUNCTION

    def test_for_function_method(self):
        """Region.for_function() with class creates method-level regions."""
        region = Region.for_function("src/main.py", "add", class_name="Calculator")

        assert region.file_path == "src/main.py"
        assert region.func_name == "add"
        assert region.class_name == "Calculator"
        assert region.level == RegionLevel.FUNCTION

    def test_for_lines(self):
        """Region.for_lines() creates line-range regions."""
        region = Region.for_lines("src/main.py", 10, 20)

        assert region.file_path == "src/main.py"
        assert region.line_start == 10
        assert region.line_end == 20
        assert region.level == RegionLevel.LINES

    def test_for_lines_with_context(self):
        """Region.for_lines() with function/class context."""
        region = Region.for_lines("src/main.py", 10, 20, func_name="compute", class_name="Calculator")

        assert region.file_path == "src/main.py"
        assert region.func_name == "compute"
        assert region.class_name == "Calculator"
        assert region.line_start == 10
        assert region.line_end == 20


class TestRegionValidation:
    """Tests for Region validation."""

    def test_empty_file_path_raises(self):
        """Empty file path raises ValueError."""
        with pytest.raises(ValueError, match="file_path cannot be empty"):
            Region(file_path="")

    def test_inverted_line_range_raises(self):
        """line_start > line_end raises ValueError."""
        with pytest.raises(ValueError, match="line_start.*cannot be greater than"):
            Region(file_path="src/main.py", line_start=20, line_end=10)

    def test_negative_line_number_raises(self):
        """Negative line numbers raise ValueError."""
        with pytest.raises(ValueError, match="line_start must be >= 1"):
            Region(file_path="src/main.py", line_start=0, line_end=10)


class TestRegionHashability:
    """Tests for Region hashability (use in sets/dicts)."""

    def test_regions_are_hashable(self):
        """Regions can be used as dict keys and set members."""
        region = Region.for_file("src/main.py")

        # Should not raise
        region_set = {region}
        region_dict = {region: "value"}

        assert region in region_set
        assert region_dict[region] == "value"

    def test_equal_regions_same_hash(self):
        """Equal regions have the same hash."""
        region1 = Region.for_function("src/main.py", "compute")
        region2 = Region.for_function("src/main.py", "compute")

        assert region1 == region2
        assert hash(region1) == hash(region2)

    def test_different_regions_different(self):
        """Different regions are not equal."""
        region1 = Region.for_function("src/main.py", "compute")
        region2 = Region.for_function("src/main.py", "other")

        assert region1 != region2
