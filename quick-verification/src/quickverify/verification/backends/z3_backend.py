"""Z3Backend — pure constraint-satisfaction verification backend.

This is the "cheap" tier in the CEGAR escalation chain (Z3 → angr → KLEE).
It checks constraints directly using Z3's SMT solver without symbolic execution.
No path exploration — the caller must encode the property as a formula.

Use cases:
  - Quick satisfiability checks before committing to angr's heavier analysis
  - Validating counterexamples from other backends (spurious CE detection)
  - Simple arithmetic/logic properties that don't need control-flow exploration
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from ..backend import IVerificationBackend
from ..types import (
    Budget,
    Counterexample,
    Predicate,
    SymbolSpec,
    VerificationRequest,
    VerificationResult,
)


# Constraint mini-DSL regex (same syntax as run_angr.py)
_CMP_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(==|!=|<=|>=|<|>)\s*([A-Za-z_]\w*|0x[0-9a-fA-F]+|-?\d+)\s*$"
)


class Z3Backend(IVerificationBackend):
    """In-process Z3 constraint solver backend.

    Does NOT perform symbolic execution. Instead, it:
      1. Creates Z3 bitvector variables for each SymbolSpec
      2. Adds all constraints from the request
      3. Adds a "find" condition (e.g., assertion_failed ⇒ encoded as reachability formula)
      4. Checks satisfiability within the budget timeout

    For ``find.kind == "return_value"``, the ``find.value`` should be a constraint
    string in the mini-DSL (e.g., ``"result<0"``).

    For ``find.kind == "assertion_failed"`` or ``"custom"``, ``find.value`` should be
    a constraint string encoding the failure condition.
    """

    def supports(self) -> dict:
        return {
            "input": "constraints",
            "constraints": ["mini-dsl"],
            "find": ["return_value", "assertion_failed", "custom"],
            "avoid": [],
        }

    def verify(self, req: VerificationRequest) -> VerificationResult:
        try:
            import z3
        except ImportError:
            return VerificationResult(
                status="error",
                stats={},
                logs="z3-solver not installed. pip install z3-solver",
            )

        t0 = time.time()
        timeout_ms = req.budget.timeout_s * 1000

        # 1. Create symbolic variables
        symmap: Dict[str, z3.BitVecRef] = {}
        for s in req.symbols:
            bits = s.bits
            if s.kind == "bool":
                bits = 1
            elif s.kind == "bytes":
                bits = (s.length or 1) * 8
            symmap[s.name] = z3.BitVec(s.name, bits)

        solver = z3.Solver()
        solver.set("timeout", timeout_ms)

        # 2. Add constraints from request
        try:
            for c in req.constraints:
                expr = self._parse_constraint(c, symmap)
                if expr is not None:
                    solver.add(expr)
        except ValueError as e:
            return VerificationResult(
                status="error",
                stats={"elapsed_s": time.time() - t0},
                logs=f"Constraint parse error: {e}",
            )

        # 3. Add find-condition (the property to violate / satisfy)
        try:
            find_expr = self._parse_find(req.find, symmap)
            if find_expr is not None:
                solver.add(find_expr)
        except ValueError as e:
            return VerificationResult(
                status="error",
                stats={"elapsed_s": time.time() - t0},
                logs=f"Find-condition parse error: {e}",
            )

        # 4. Check satisfiability
        result = solver.check()
        elapsed = time.time() - t0

        stats = {
            "elapsed_s": elapsed,
            "solver_result": str(result),
            "num_constraints": len(req.constraints),
            "num_symbols": len(req.symbols),
        }

        if result == z3.sat:
            model = solver.model()
            concrete: Dict[str, Any] = {}
            for s in req.symbols:
                var = symmap[s.name]
                val = model.evaluate(var, model_completion=True)
                try:
                    concrete[s.name] = val.as_long()
                except Exception:
                    concrete[s.name] = str(val)

            ce = Counterexample(
                model=concrete,
                trace=[],  # Z3 has no execution trace
                path_constraints=[str(c) for c in solver.assertions()],
                notes={"backend": "z3", "solver_stats": str(solver.statistics())},
            )
            return VerificationResult(status="ce_found", counterexample=ce, stats=stats)

        if result == z3.unsat:
            # UNSAT means the property CANNOT be violated under these constraints
            # This is a stronger result than "no_ce_within_budget"
            stats["proof"] = "unsat"
            return VerificationResult(status="no_ce_within_budget", stats=stats)

        # z3.unknown — typically timeout
        return VerificationResult(
            status="no_ce_within_budget",
            stats=stats,
            logs=f"Z3 returned unknown (likely timeout): {solver.reason_unknown()}",
        )

    # -- Constraint parsing --------------------------------------------------

    def _parse_constraint(self, constraint: str, symmap: Dict[str, Any]) -> Any:
        """Parse a mini-DSL constraint into a Z3 expression."""
        import z3

        m = _CMP_RE.match(constraint)
        if not m:
            raise ValueError(f"Cannot parse constraint: {constraint!r}")
        lhs_name, op, rhs_str = m.group(1), m.group(2), m.group(3)
        lhs = self._resolve_atom(lhs_name, symmap)
        rhs = self._resolve_atom(rhs_str, symmap)

        # Width alignment
        lhs, rhs = self._align_widths(lhs, rhs)

        ops = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b,
               "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
               "<": lambda a, b: z3.ULT(a, b), ">": lambda a, b: z3.UGT(a, b)}
        if op not in ops:
            raise ValueError(f"Unknown operator: {op}")
        return ops[op](lhs, rhs)

    def _parse_find(self, pred: Predicate, symmap: Dict[str, Any]) -> Any:
        """Convert a find-predicate into a Z3 expression."""
        if pred.kind in ("return_value", "assertion_failed", "custom"):
            if isinstance(pred.value, str):
                return self._parse_constraint(pred.value, symmap)
        # addr_reached can't be handled by Z3 (need symbolic execution)
        if pred.kind == "addr_reached":
            raise ValueError(
                "Z3Backend cannot handle 'addr_reached' predicates. "
                "Use angr or another symbolic execution backend."
            )
        return None

    def _resolve_atom(self, token: str, symmap: Dict[str, Any]) -> Any:
        import z3

        token = token.strip()
        if token in symmap:
            return symmap[token]
        if token.lower().startswith("0x"):
            return z3.BitVecVal(int(token, 16), 64)
        if re.fullmatch(r"-?\d+", token):
            return z3.BitVecVal(int(token), 64)
        raise ValueError(f"Unknown atom: {token!r}")

    def _align_widths(self, a: Any, b: Any) -> tuple:
        import z3

        aw = a.size()
        bw = b.size()
        if aw == bw:
            return a, b
        w = max(aw, bw)
        if aw < w:
            a = z3.ZeroExt(w - aw, a)
        if bw < w:
            b = z3.ZeroExt(w - bw, b)
        return a, b
