"""
Verification orchestrator: CEGAR loop with budget escalation.

Chains cheap verification (Z3) → expensive (angr) with escalating
budgets. Optionally checks for spurious counterexamples.

Decision: D-016
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from verification.backend import VerificationBackend
from verification.types import (
    Budget,
    Counterexample,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)

LOG = logging.getLogger("verification.orchestrator")

# Default budget escalation presets: 10s → 30s → 120s
DEFAULT_BUDGET_PRESETS = [
    Budget(timeout_s=10, max_states=10_000, max_path_len=100, max_loop_iters=3),
    Budget(timeout_s=30, max_states=50_000, max_path_len=200, max_loop_iters=5),
    Budget(timeout_s=120, max_states=200_000, max_path_len=500, max_loop_iters=10),
]


@dataclass
class OrchestratorResult:
    """Aggregated result from the orchestrator's CEGAR loop."""

    status: str = "unknown"
    counterexample: Optional[Counterexample] = None
    iterations: int = 0
    total_elapsed_s: float = 0.0
    backend_results: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "counterexample": self.counterexample.to_dict() if self.counterexample else None,
            "iterations": self.iterations,
            "total_elapsed_s": self.total_elapsed_s,
            "backend_results": self.backend_results,
            "logs": self.logs,
        }


class VerificationOrchestrator:
    """
    CEGAR-style orchestrator with budget escalation.

    Runs verification across one or more backends with progressively
    larger budgets. Supports spurious CE checking via a callback.

    Usage::

        orch = VerificationOrchestrator(backend=z3_backend)
        result = await orch.run(request)
    """

    def __init__(
        self,
        backend: VerificationBackend,
        budget_presets: Optional[List[Budget]] = None,
        spurious_checker: Optional[Callable[[Counterexample], bool]] = None,
        max_iterations: int = 5,
    ) -> None:
        self._backend = backend
        self._budgets = budget_presets or DEFAULT_BUDGET_PRESETS
        self._spurious_checker = spurious_checker
        self._max_iterations = max_iterations

    async def run(self, request: VerificationRequest) -> OrchestratorResult:
        """
        Run CEGAR loop with budget escalation.

        For each budget preset:
        1. Run verification with that budget
        2. If CE found, check if spurious (if checker provided)
        3. If spurious, escalate budget and retry
        4. If confirmed CE or budget exhausted, return
        """
        t0 = time.monotonic()
        result = OrchestratorResult()

        for i, budget in enumerate(self._budgets):
            if i >= self._max_iterations:
                break

            result.iterations = i + 1
            req = VerificationRequest(
                target_binary=request.target_binary,
                entry=request.entry,
                symbols=request.symbols,
                constraints=request.constraints,
                find_kind=request.find_kind,
                find_value=request.find_value,
                avoid_kind=request.avoid_kind,
                avoid_value=request.avoid_value,
                budget=budget,
                metadata=request.metadata,
                notes=request.notes,
            )

            LOG.info(
                "Orchestrator iteration %d: budget timeout=%ds, max_states=%d",
                i + 1, budget.timeout_s, budget.max_states,
            )

            vr = await self._backend.verify(req)
            result.backend_results.append(vr.to_dict())

            if vr.status == VerificationStatus.CE_FOUND and vr.counterexample:
                # Check if spurious
                if self._spurious_checker and self._spurious_checker(vr.counterexample):
                    result.logs.append(
                        f"Iteration {i + 1}: spurious CE, escalating budget"
                    )
                    LOG.info("Spurious CE at iteration %d, escalating", i + 1)
                    continue

                # Confirmed CE
                result.status = "ce_found"
                result.counterexample = vr.counterexample
                result.total_elapsed_s = time.monotonic() - t0
                return result

            if vr.status == VerificationStatus.ERROR:
                result.logs.append(f"Iteration {i + 1}: error — {vr.logs}")
                result.status = "error"
                result.total_elapsed_s = time.monotonic() - t0
                return result

            # no_ce_within_budget — continue to next budget
            result.logs.append(
                f"Iteration {i + 1}: no CE within budget (timeout={budget.timeout_s}s)"
            )

        # All budgets exhausted without finding a CE
        result.status = "bounded_safe"
        result.total_elapsed_s = time.monotonic() - t0
        return result

    async def run_multi_backend(
        self,
        request: VerificationRequest,
        backends: List[VerificationBackend],
    ) -> OrchestratorResult:
        """
        Run verification across multiple backends (cheap → expensive).

        Stops at the first backend that finds a confirmed CE or errors.
        Falls through to the next backend on no_ce_within_budget.
        """
        t0 = time.monotonic()
        result = OrchestratorResult()

        for i, backend in enumerate(backends):
            result.iterations = i + 1
            LOG.info("Multi-backend: trying backend %d (%s)", i + 1, type(backend).__name__)

            vr = await backend.verify(request)
            result.backend_results.append(vr.to_dict())

            if vr.status == VerificationStatus.CE_FOUND and vr.counterexample:
                if self._spurious_checker and self._spurious_checker(vr.counterexample):
                    result.logs.append(
                        f"Backend {i + 1}: spurious CE, trying next backend"
                    )
                    continue

                result.status = "ce_found"
                result.counterexample = vr.counterexample
                result.total_elapsed_s = time.monotonic() - t0
                return result

            if vr.status == VerificationStatus.ERROR:
                result.logs.append(f"Backend {i + 1}: error — {vr.logs}")
                # Don't stop on error — try next backend
                continue

            result.logs.append(
                f"Backend {i + 1}: no CE within budget"
            )

        result.status = "bounded_safe"
        result.total_elapsed_s = time.monotonic() - t0
        return result
