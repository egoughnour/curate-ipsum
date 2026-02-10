"""
Hierarchical code region model for curate-ipsum.

Regions identify code locations at various granularities:
- File level: file:src/main.py
- Class level: file:src/main.py::class:Calculator
- Function level: file:src/main.py::func:compute
- Line range: file:src/main.py::lines:10-25

Regions support containment queries (does region A contain region B?)
and overlap detection for aggregating metrics across related code.
"""

from regions.models import Region, RegionLevel, normalize_region_id, region_to_id

__all__ = [
    "Region",
    "RegionLevel",
    "normalize_region_id",
    "region_to_id",
]
