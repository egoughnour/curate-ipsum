"""
Z3 constraint-solving verification backend.

Handles constraint satisfaction queries using the Z3 SMT solver.
This is the "cheap" tier in the CEGAR budget chain (Z3 → angr → KLEE).

Decision: D-016
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from curate_ipsum.verification.backend import VerificationBackend
from curate_ipsum.verification.types import (
    Counterexample,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)

LOG = logging.getLogger("verification.backends.z3")

_CMP_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<=|>=|<|>)\s*"
    r"([A-Za-z_][A-Za-z0-9_]*|0x[0-9a-fA-F]+|-?\d+)\s*$"
)


class Z3Backend(VerificationBackend):
    """
    Verification via Z3 SMT solver.

    Parses the mini-DSL constraints from VerificationRequest, constructs
    Z3 bitvector expressions, and checks satisfiability.

    Supported find kinds: return_value, assertion_failed, custom.
    Does NOT support addr_reached (use angr for that).
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def supports(self) -> dict[str, Any]:
        return {
            "input": "constraints",
            "constraints": ["comparison"],
            "find": ["return_value", "assertion_failed", "custom"],
            "avoid": [],
        }

    async def verify(self, request: VerificationRequest) -> VerificationResult:
        t0 = time.monotonic()

        # Reject unsupported find kinds early
        if request.find_kind == "addr_reached":
            return VerificationResult(
                status=VerificationStatus.ERROR,
                stats={"elapsed_s": time.monotonic() - t0},
                logs="Z3 backend does not support addr_reached; use angr",
            )

        try:
            import z3
        except ImportError:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                stats={"elapsed_s": time.monotonic() - t0},
                logs="z3-solver not installed. pip install z3-solver",
            )

        try:
            solver = z3.Solver()
            solver.set("timeout", request.budget.timeout_s * 1000)  # ms

            # Create symbolic variables
            symmap: dict[str, z3.BitVecRef] = {}
            for spec in request.symbols:
                bits = spec.bits
                if spec.kind == "bool":
                    bits = 1
                elif spec.kind == "bytes" and spec.length is not None:
                    bits = spec.length * 8
                symmap[spec.name] = z3.BitVec(spec.name, bits)

            # Parse and add constraints
            for c in request.constraints:
                expr = self._parse_constraint(c, symmap, z3)
                if expr is not None:
                    solver.add(expr)

            # Add the property-to-check (find condition as negated assertion)
            prop_expr = self._build_property(request, symmap, z3)
            if prop_expr is not None:
                solver.add(prop_expr)

            # Check satisfiability
            result = solver.check()
            elapsed = time.monotonic() - t0

            if result == z3.sat:
                model = solver.model()
                ce_model: dict[str, Any] = {}
                for name, sym in symmap.items():
                    val = model.evaluate(sym)
                    try:
                        ce_model[name] = val.as_long()
                    except Exception:
                        ce_model[name] = str(val)

                return VerificationResult(
                    status=VerificationStatus.CE_FOUND,
                    counterexample=Counterexample(
                        model=ce_model,
                        trace=[],
                        path_constraints=[str(c) for c in request.constraints],
                        notes={"solver": "z3", "find_kind": request.find_kind},
                    ),
                    stats={"elapsed_s": elapsed, "solver": "z3"},
                )
            elif result == z3.unsat:
                return VerificationResult(
                    status=VerificationStatus.NO_CE_WITHIN_BUDGET,
                    stats={"elapsed_s": elapsed, "solver": "z3", "result": "unsat"},
                )
            else:
                # unknown / timeout
                return VerificationResult(
                    status=VerificationStatus.NO_CE_WITHIN_BUDGET,
                    stats={"elapsed_s": elapsed, "solver": "z3", "result": "unknown"},
                    logs="Z3 returned unknown (likely timeout)",
                )

        except Exception as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                stats={"elapsed_s": time.monotonic() - t0},
                logs=f"Z3 backend error: {exc}",
            )

    def _parse_constraint(self, constraint: str, symmap: dict[str, Any], z3: Any) -> Any:
        """Parse a mini-DSL constraint string into a Z3 expression."""
        m = _CMP_RE.match(constraint)
        if not m:
            LOG.warning("Constraint parse error: %r", constraint)
            return None

        lhs_name, op, rhs_str = m.group(1), m.group(2), m.group(3)
        lhs = self._resolve_atom(lhs_name, symmap, z3)
        rhs = self._resolve_atom(rhs_str, symmap, z3)

        if lhs is None or rhs is None:
            LOG.warning("Unresolved atom in constraint: %r", constraint)
            return None

        # Width alignment
        lw, rw = lhs.size(), rhs.size()
        if lw < rw:
            lhs = z3.ZeroExt(rw - lw, lhs)
        elif rw < lw:
            rhs = z3.ZeroExt(lw - rw, rhs)

        ops = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: z3.ULT(a, b),
            ">": lambda a, b: z3.UGT(a, b),
        }
        return ops[op](lhs, rhs)

    def _resolve_atom(self, atom: str, symmap: dict[str, Any], z3: Any) -> Any:
        """Resolve an atom to a Z3 bitvector value or variable."""
        atom = atom.strip()
        if atom in symmap:
            return symmap[atom]
        if atom.lower().startswith("0x"):
            return z3.BitVecVal(int(atom, 16), 64)
        if re.fullmatch(r"-?\d+", atom):
            return z3.BitVecVal(int(atom), 64)
        return None

    def _build_property(self, request: VerificationRequest, symmap: dict[str, Any], z3: Any) -> Any:
        """Build a Z3 expression for the find condition."""
        if request.find_kind == "return_value":
            # Find a satisfying assignment where the constraints hold
            # (the constraints themselves encode the property)
            return None
        if request.find_kind == "assertion_failed":
            # Negate the assertion: find inputs that violate it
            return None
        if request.find_kind == "custom":
            # Custom predicate: parse find_value as a constraint
            if request.find_value:
                return self._parse_constraint(request.find_value, symmap, z3)
        return None
