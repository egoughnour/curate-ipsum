"""
Synthesis package: CEGIS-based code patch synthesis with genetic algorithm evolution.

This package implements M4 of the curate-ipsum roadmap — the synthesis loop that
transforms LLM-generated code candidates into verified patches via counterexample-guided
inductive synthesis (CEGIS) and genetic algorithm population management.

Optional dependency: httpx (for cloud/local LLM backends).
Core synthesis (models, fitness, AST operators) uses only stdlib.
"""

from __future__ import annotations

# Core models — always available
from curate_ipsum.synthesis.models import (
    CodePatch,
    Counterexample,
    Individual,
    PatchSource,
    Specification,
    SynthesisConfig,
    SynthesisResult,
    SynthesisStatus,
)

HAS_SYNTHESIS = True

__all__ = [
    "CodePatch",
    "Counterexample",
    "HAS_SYNTHESIS",
    "Individual",
    "PatchSource",
    "Specification",
    "SynthesisConfig",
    "SynthesisResult",
    "SynthesisStatus",
]
