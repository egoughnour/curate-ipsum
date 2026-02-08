"""Verification data model — frozen dataclasses for request/response cycle.

Ported from angr_adapter_baseline/verification/types.py with no breaking changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass(frozen=True)
class Budget:
    """Hard limits on a single verification run."""
    timeout_s: int = 10
    max_states: int = 256
    max_path_len: int = 200
    max_loop_iters: int = 8

    def escalate(self, factor: int = 2) -> "Budget":
        """Return a new Budget with all limits multiplied by *factor*."""
        return Budget(
            timeout_s=self.timeout_s * factor,
            max_states=self.max_states * factor,
            max_path_len=self.max_path_len * factor,
            max_loop_iters=self.max_loop_iters * factor,
        )


@dataclass(frozen=True)
class SymbolSpec:
    """Specification for a single symbolic variable."""
    name: str
    bits: int
    kind: Literal["int", "bool", "bytes"]
    length: Optional[int] = None  # for kind=="bytes"


@dataclass(frozen=True)
class Predicate:
    """Search predicate for find/avoid."""
    kind: Literal["addr_reached", "addr_avoided", "return_value", "assertion_failed", "custom"]
    value: Any


@dataclass(frozen=True)
class VerificationTarget:
    """Locates a binary + entry point."""
    binary_name: str
    entry: str  # symbol name or "0x..." hex address
    calling_convention: Optional[str] = None


@dataclass(frozen=True)
class VerificationRequest:
    """Complete verification job specification."""
    target: VerificationTarget
    symbols: List[SymbolSpec]
    constraints: List[str]  # mini-DSL: "x>=0", "x<=2000", "x==y"
    find: Predicate
    budget: Budget = field(default_factory=Budget)
    avoid: Optional[Predicate] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict (matches schemas/request.schema.json)."""
        return {
            "target": {
                "binary_name": self.target.binary_name,
                "entry": self.target.entry,
                "calling_convention": self.target.calling_convention,
            },
            "symbols": [
                {"name": s.name, "bits": s.bits, "kind": s.kind, "length": s.length}
                for s in self.symbols
            ],
            "constraints": list(self.constraints),
            "find": {"kind": self.find.kind, "value": self.find.value},
            "avoid": (
                {"kind": self.avoid.kind, "value": self.avoid.value}
                if self.avoid
                else None
            ),
            "budget": {
                "timeout_s": self.budget.timeout_s,
                "max_states": self.budget.max_states,
                "max_path_len": self.budget.max_path_len,
                "max_loop_iters": self.budget.max_loop_iters,
            },
            "metadata": dict(self.metadata),
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, d: dict) -> "VerificationRequest":
        t = d["target"]
        return cls(
            target=VerificationTarget(
                binary_name=t["binary_name"],
                entry=t["entry"],
                calling_convention=t.get("calling_convention"),
            ),
            symbols=[
                SymbolSpec(name=s["name"], bits=s["bits"], kind=s["kind"], length=s.get("length"))
                for s in d.get("symbols", [])
            ],
            constraints=d.get("constraints", []),
            find=Predicate(kind=d["find"]["kind"], value=d["find"]["value"]),
            avoid=(
                Predicate(kind=d["avoid"]["kind"], value=d["avoid"]["value"])
                if d.get("avoid")
                else None
            ),
            budget=Budget(**d["budget"]) if "budget" in d else Budget(),
            metadata=d.get("metadata", {}),
            notes=d.get("notes"),
        )


@dataclass(frozen=True)
class Counterexample:
    """A concrete counterexample found by a verification backend."""
    model: Dict[str, Any]         # concrete inputs: {"x": 337, "y": 1000}
    trace: List[Dict[str, Any]]   # BBL addresses or steps visited
    path_constraints: List[str]   # serialized path constraints
    notes: Dict[str, Any]         # backend-specific extras


@dataclass(frozen=True)
class VerificationResult:
    """Result of a single verification run."""
    status: Literal["ce_found", "no_ce_within_budget", "error"]
    counterexample: Optional[Counterexample] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    logs: Optional[str] = None

    def to_json(self) -> dict:
        ce = None
        if self.counterexample:
            ce = {
                "model": self.counterexample.model,
                "trace": self.counterexample.trace,
                "path_constraints": self.counterexample.path_constraints,
                "notes": self.counterexample.notes,
            }
        return {
            "status": self.status,
            "counterexample": ce,
            "stats": self.stats,
            "logs": self.logs,
        }

    @classmethod
    def from_json(cls, d: dict) -> "VerificationResult":
        ce = None
        if d.get("counterexample"):
            c = d["counterexample"]
            ce = Counterexample(
                model=c.get("model", {}),
                trace=c.get("trace", []),
                path_constraints=c.get("path_constraints", []),
                notes=c.get("notes", {}),
            )
        return cls(
            status=d.get("status", "error"),
            counterexample=ce,
            stats=d.get("stats", {}),
            logs=d.get("logs"),
        )
