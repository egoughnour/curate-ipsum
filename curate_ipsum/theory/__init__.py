"""
Theory management module for curate-ipsum.

This module provides the TheoryManager class that wraps py-brs operations
with curate-ipsum-specific logic for managing synthesis theories.

Submodules:
    - assertions: Typed assertion model with contradiction detection
    - provenance: Append-only causal chain for belief evolution
    - rollback: State recovery API over CASStore world versioning
    - failure_analyzer: Heuristic failure classification for synthesis
"""

from theory.manager import TheoryManager

__all__ = ["TheoryManager"]
