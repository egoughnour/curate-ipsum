"""
Core data models for the synthesis loop.

All models use Pydantic for validation and serialization, matching the
project's existing pattern (see models.py in the root).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class SynthesisStatus(str, Enum):
    """Outcome of a synthesis run."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class PatchSource(str, Enum):
    """How a code patch was produced."""

    LLM = "llm"
    CROSSOVER = "crossover"
    MUTATION = "mutation"
    SEED = "seed"


class LLMBackend(str, Enum):
    """Which LLM backend to use."""

    CLOUD = "cloud"
    LOCAL = "local"
    MOCK = "mock"


@dataclass
class SynthesisConfig:
    """Configuration for a synthesis run."""

    # Population / GA
    population_size: int = 20
    max_iterations: int = 100
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elite_ratio: float = 0.1
    entropy_threshold: float = 1.0

    # LLM
    llm_backend: str = "mock"  # "cloud", "local", "mock"
    llm_model: str = "codellama:7b"
    temperature: float = 0.8
    top_k: int = 10

    # Fitness weights
    ce_weight: float = 0.4
    spec_weight: float = 0.5
    complexity_weight: float = 0.1

    # Timeouts
    test_timeout_seconds: float = 30.0
    synthesis_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be 0-1, got {self.mutation_rate}")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be 0-1, got {self.crossover_rate}")
        if not 0.0 <= self.elite_ratio <= 1.0:
            raise ValueError(f"elite_ratio must be 0-1, got {self.elite_ratio}")
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


@dataclass
class Individual:
    """A candidate patch in the genetic algorithm population."""

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    code: str = ""
    fitness: float = 0.0
    lineage: list[str] = field(default_factory=list)  # Parent IDs
    generation: int = 0
    source: PatchSource = PatchSource.SEED
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if the code is syntactically valid Python."""
        try:
            ast.parse(self.code)
            return True
        except SyntaxError:
            return False


@dataclass
class CodePatch:
    """A code patch produced by synthesis."""

    code: str
    source: PatchSource = PatchSource.LLM
    diff: str = ""
    region_id: str = ""
    original_code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "source": self.source.value,
            "diff": self.diff,
            "region_id": self.region_id,
            "metadata": self.metadata,
        }


@dataclass
class Specification:
    """What a synthesized patch must satisfy."""

    target_region: str = ""  # Region ID
    original_code: str = ""  # Code being replaced
    surviving_mutant_ids: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    mutation_command: str = ""
    working_directory: str = ""
    # Assertions from M3 belief revision
    assertion_ids: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    context_code: str = ""  # Surrounding code for LLM prompt context
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Counterexample:
    """A counterexample that a candidate patch fails on."""

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    input_values: dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    actual_output: Any = None
    mutant_id: str = ""
    error_message: str = ""
    test_command: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_values": self.input_values,
            "expected_output": str(self.expected_output),
            "actual_output": str(self.actual_output),
            "mutant_id": self.mutant_id,
            "error_message": self.error_message,
            "test_command": self.test_command,
        }


@dataclass
class SynthesisResult:
    """Outcome of a synthesis run."""

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    status: SynthesisStatus = SynthesisStatus.FAILED
    patch: CodePatch | None = None
    iterations: int = 0
    counterexamples_resolved: int = 0
    duration_ms: int = 0
    fitness_history: list[float] = field(default_factory=list)
    final_entropy: float = 0.0
    total_candidates_evaluated: int = 0
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "patch": self.patch.to_dict() if self.patch else None,
            "iterations": self.iterations,
            "counterexamples_resolved": self.counterexamples_resolved,
            "duration_ms": self.duration_ms,
            "fitness_history": self.fitness_history,
            "final_entropy": self.final_entropy,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "error_message": self.error_message,
        }
