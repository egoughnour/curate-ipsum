"""
Core data models for the verification subsystem.

Uses dataclasses consistent with synthesis.models pattern (D-012).
All models support dict serialization for JSON exchange with Docker runners.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

LOG = logging.getLogger("verification.types")


class VerificationStatus(StrEnum):
    """Outcome of a verification run."""

    CE_FOUND = "ce_found"
    NO_CE_WITHIN_BUDGET = "no_ce_within_budget"
    ERROR = "error"


@dataclass
class Budget:
    """Resource bounds for a single verification run.

    Mirrors the budget schema in angr_adapter_baseline/schemas/request.schema.json.
    """

    timeout_s: int = 30
    max_states: int = 50_000
    max_path_len: int = 200
    max_loop_iters: int = 5

    def escalate(self, factor: float = 2.0) -> "Budget":
        """Return a new Budget with all limits scaled by *factor*."""
        return Budget(
            timeout_s=int(self.timeout_s * factor),
            max_states=int(self.max_states * factor),
            max_path_len=int(self.max_path_len * factor),
            max_loop_iters=int(self.max_loop_iters * factor),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_s": self.timeout_s,
            "max_states": self.max_states,
            "max_path_len": self.max_path_len,
            "max_loop_iters": self.max_loop_iters,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Budget":
        return cls(
            timeout_s=int(d["timeout_s"]),
            max_states=int(d["max_states"]),
            max_path_len=int(d["max_path_len"]),
            max_loop_iters=int(d["max_loop_iters"]),
        )


@dataclass
class SymbolSpec:
    """A symbolic variable declaration for the verification request."""

    name: str
    kind: str = "int"  # "int", "bool", "bytes"
    bits: int = 64
    length: int | None = None  # Required for kind="bytes"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "kind": self.kind, "bits": self.bits}
        if self.length is not None:
            d["length"] = self.length
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SymbolSpec":
        return cls(
            name=d["name"],
            kind=d.get("kind", "int"),
            bits=int(d.get("bits", 64)),
            length=d.get("length"),
        )


@dataclass
class VerificationRequest:
    """A complete verification request, matching request.schema.json."""

    target_binary: str
    entry: str
    symbols: list[SymbolSpec] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    find_kind: str = "addr_reached"
    find_value: str = ""
    avoid_kind: str | None = None
    avoid_value: str | None = None
    budget: Budget = field(default_factory=Budget)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "target": {
                "binary_name": self.target_binary,
                "entry": self.entry,
                "calling_convention": None,
            },
            "symbols": [s.to_dict() for s in self.symbols],
            "constraints": self.constraints,
            "find": {"kind": self.find_kind, "value": self.find_value},
            "budget": self.budget.to_dict(),
            "metadata": self.metadata,
            "notes": self.notes,
        }
        if self.avoid_kind:
            d["avoid"] = {"kind": self.avoid_kind, "value": self.avoid_value}
        else:
            d["avoid"] = None
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VerificationRequest":
        target = d["target"]
        find = d["find"]
        avoid = d.get("avoid")
        return cls(
            target_binary=target["binary_name"],
            entry=target["entry"],
            symbols=[SymbolSpec.from_dict(s) for s in d.get("symbols", [])],
            constraints=d.get("constraints", []),
            find_kind=find["kind"],
            find_value=str(find["value"]),
            avoid_kind=avoid["kind"] if avoid else None,
            avoid_value=str(avoid["value"]) if avoid else None,
            budget=Budget.from_dict(d["budget"]),
            metadata=d.get("metadata", {}),
            notes=d.get("notes"),
        )


@dataclass
class Counterexample:
    """A concrete counterexample found by a verification backend."""

    model: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    path_constraints: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "trace": self.trace,
            "path_constraints": self.path_constraints,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Counterexample":
        return cls(
            model=d.get("model", {}),
            trace=d.get("trace", []),
            path_constraints=d.get("path_constraints", []),
            notes=d.get("notes", {}),
        )


@dataclass
class VerificationResult:
    """Result from a verification backend, matching response.schema.json."""

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    status: VerificationStatus = VerificationStatus.NO_CE_WITHIN_BUDGET
    counterexample: Counterexample | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    logs: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "counterexample": self.counterexample.to_dict() if self.counterexample else None,
            "stats": self.stats,
            "logs": self.logs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VerificationResult":
        ce_data = d.get("counterexample")
        return cls(
            id=d.get("id", str(uuid4())[:8]),
            status=VerificationStatus(d["status"]),
            counterexample=Counterexample.from_dict(ce_data) if ce_data else None,
            stats=d.get("stats", {}),
            logs=d.get("logs"),
        )
