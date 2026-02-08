"""VerificationOrchestrator — CEGAR-style loop with budget escalation.

Coordinates: GraphStore → HarnessBuilder → IVerificationBackend → result evaluation.

Escalation chain:
  Attempt 0: 10s / 256 states / 200 path / 8 loops
  Attempt 1: 30s / 512 states / 400 path / 16 loops
  Attempt 2: 120s / 1024 states / 800 path / 32 loops
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from .backend import IVerificationBackend
from .types import (
    Budget,
    Predicate,
    SymbolSpec,
    VerificationRequest,
    VerificationResult,
    VerificationTarget,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PropertySpec:
    """What to verify — a high-level property specification."""
    description: str
    target_hint: Optional[str] = None
    constraint_templates: Optional[List[str]] = None
    find_kind: str = "addr_reached"
    find_value: str = "violation"
    avoid_kind: Optional[str] = "addr_avoided"
    avoid_value: Optional[str] = "ok_exit"
    symbols: Optional[List[SymbolSpec]] = None


@dataclass
class OrchestratorResult:
    """Outcome of a full CEGAR-style verification attempt."""
    verdict: Literal["ce_confirmed", "bounded_safe", "error", "spurious_all"]
    final_result: Optional[VerificationResult] = None
    attempts: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


# Budget escalation presets (indexed by attempt number)
BUDGET_PRESETS = [
    Budget(timeout_s=10, max_states=256, max_path_len=200, max_loop_iters=8),
    Budget(timeout_s=30, max_states=512, max_path_len=400, max_loop_iters=16),
    Budget(timeout_s=120, max_states=1024, max_path_len=800, max_loop_iters=32),
]


class VerificationOrchestrator:
    """Runs a CEGAR-style verification loop with budget escalation.

    The orchestrator:
      1. Selects a target slice (via graph_store or property spec)
      2. Builds a harness binary (via harness_builder)
      3. Dispatches to the verification backend (Z3, angr, etc.)
      4. Evaluates the result: CE → check spurious → refine or accept
      5. Escalates budget on retry

    Args:
        backend: Verification backend (or a list for multi-backend escalation).
        harness_builder: Builds compiled binaries from target specs.
        max_refinements: Maximum CEGAR iterations.
        budget_presets: Budget escalation sequence.
        spurious_checker: Optional callable (result, spec) → bool.
    """

    def __init__(
        self,
        backend: IVerificationBackend,
        harness_builder: Optional[Any] = None,
        max_refinements: int = 3,
        budget_presets: Optional[List[Budget]] = None,
        spurious_checker: Optional[Callable[[VerificationResult, PropertySpec], bool]] = None,
    ):
        self.backend = backend
        self.harness_builder = harness_builder
        self.max_refinements = max_refinements
        self.budgets = budget_presets or list(BUDGET_PRESETS)
        self.spurious_checker = spurious_checker

    def verify_property(self, spec: PropertySpec) -> OrchestratorResult:
        """Run the full CEGAR loop for a property specification."""
        history: List[Dict[str, Any]] = []

        for attempt in range(self.max_refinements):
            budget = self._budget_for_attempt(attempt)
            log.info("Attempt %d/%d with budget %s", attempt + 1, self.max_refinements, budget)

            # Build harness if builder available
            target = self._resolve_target(spec)

            # Create verification request
            request = VerificationRequest(
                target=target,
                symbols=spec.symbols or [],
                constraints=spec.constraint_templates or [],
                find=Predicate(kind=spec.find_kind, value=spec.find_value),
                avoid=(
                    Predicate(kind=spec.avoid_kind, value=spec.avoid_value)
                    if spec.avoid_kind and spec.avoid_value
                    else None
                ),
                budget=budget,
                metadata={"property": spec.description, "attempt": attempt},
            )

            # Dispatch to backend
            result = self.backend.verify(request)
            history.append({
                "attempt": attempt,
                "budget": {
                    "timeout_s": budget.timeout_s,
                    "max_states": budget.max_states,
                    "max_path_len": budget.max_path_len,
                    "max_loop_iters": budget.max_loop_iters,
                },
                "status": result.status,
                "stats": result.stats,
                "has_ce": result.counterexample is not None,
            })

            # Evaluate
            if result.status == "ce_found":
                is_spurious = False
                if self.spurious_checker:
                    is_spurious = self.spurious_checker(result, spec)
                if is_spurious:
                    log.info("Spurious CE detected — refining (attempt %d)", attempt)
                    continue
                log.info("CE confirmed on attempt %d", attempt)
                return OrchestratorResult(
                    verdict="ce_confirmed",
                    final_result=result,
                    attempts=attempt + 1,
                    history=history,
                )

            if result.status == "no_ce_within_budget":
                if attempt < self.max_refinements - 1:
                    log.info("No CE within budget — escalating (attempt %d)", attempt)
                    continue
                return OrchestratorResult(
                    verdict="bounded_safe",
                    final_result=result,
                    attempts=attempt + 1,
                    history=history,
                )

            # Error
            log.warning("Backend error on attempt %d: %s", attempt, result.logs)
            return OrchestratorResult(
                verdict="error",
                final_result=result,
                attempts=attempt + 1,
                history=history,
            )

        return OrchestratorResult(
            verdict="spurious_all",
            attempts=self.max_refinements,
            history=history,
        )

    def verify_multi_backend(
        self,
        spec: PropertySpec,
        backends: List[IVerificationBackend],
    ) -> OrchestratorResult:
        """Try backends in order (cheap → expensive). Stop at first decisive result."""
        for i, backend in enumerate(backends):
            log.info("Trying backend %d/%d: %s", i + 1, len(backends), type(backend).__name__)
            original = self.backend
            self.backend = backend
            try:
                result = self.verify_property(spec)
                if result.verdict in ("ce_confirmed", "bounded_safe"):
                    return result
            finally:
                self.backend = original
        return OrchestratorResult(verdict="error", attempts=len(backends))

    def _budget_for_attempt(self, attempt: int) -> Budget:
        if attempt < len(self.budgets):
            return self.budgets[attempt]
        last = self.budgets[-1]
        factor = 2 ** (attempt - len(self.budgets) + 1)
        return last.escalate(factor)

    def _resolve_target(self, spec: PropertySpec) -> VerificationTarget:
        """Resolve spec to a VerificationTarget, using harness_builder if available."""
        if self.harness_builder and hasattr(self.harness_builder, "build"):
            try:
                ht = self.harness_builder.build_from_spec(spec)
                return VerificationTarget(
                    binary_name=ht.binary_name,
                    entry=ht.entry,
                )
            except Exception as e:
                log.warning("HarnessBuilder failed: %s — using spec hint", e)

        return VerificationTarget(
            binary_name=spec.target_hint or "harness",
            entry=spec.find_value,
        )
