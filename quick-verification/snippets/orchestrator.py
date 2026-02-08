"""
VerificationOrchestrator — CEGAR-style loop with budget escalation.

Coordinates: GraphStore → HarnessBuilder → IVerificationBackend → result evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol


# -- Protocols for dependencies --

class VerificationBackendProtocol(Protocol):
    def verify(self, req: Any) -> Any: ...
    def supports(self) -> dict: ...

class GraphStoreProtocol(Protocol):
    def get_partition_nodes(self, property_spec: Any) -> List[str]: ...

class HarnessBuilderProtocol(Protocol):
    def build(self, node_ids: List[str]) -> Any: ...


@dataclass(frozen=True)
class Budget:
    timeout_s: int = 10
    max_states: int = 256
    max_path_len: int = 200
    max_loop_iters: int = 8


@dataclass(frozen=True)
class PropertySpec:
    """What to verify: a property name, target slice hint, constraint templates."""
    description: str
    target_hint: Optional[str] = None  # symbol name or partition ID
    constraint_templates: Optional[List[str]] = None
    find_kind: str = "addr_reached"
    find_value: str = "violation"
    avoid_kind: Optional[str] = "addr_avoided"
    avoid_value: Optional[str] = "ok_exit"


@dataclass
class OrchestratorResult:
    verdict: Literal["ce_confirmed", "bounded_safe", "error", "spurious_all"]
    final_result: Optional[Any] = None
    attempts: int = 0
    history: Optional[List[Dict[str, Any]]] = None


# Budget escalation presets (indexed by attempt number)
BUDGET_PRESETS = [
    Budget(timeout_s=10, max_states=256, max_path_len=200, max_loop_iters=8),
    Budget(timeout_s=30, max_states=512, max_path_len=400, max_loop_iters=16),
    Budget(timeout_s=120, max_states=1024, max_path_len=800, max_loop_iters=32),
]


class VerificationOrchestrator:
    """Runs a CEGAR-style verification loop with budget escalation."""

    def __init__(
        self,
        backend: VerificationBackendProtocol,
        graph_store: GraphStoreProtocol,
        harness_builder: HarnessBuilderProtocol,
        max_refinements: int = 3,
        budget_presets: Optional[List[Budget]] = None,
    ):
        self.backend = backend
        self.graph_store = graph_store
        self.harness_builder = harness_builder
        self.max_refinements = max_refinements
        self.budgets = budget_presets or BUDGET_PRESETS

    def verify_property(self, spec: PropertySpec) -> OrchestratorResult:
        """Run the full CEGAR loop for a property specification."""
        history = []

        for attempt in range(self.max_refinements):
            budget = self._budget_for_attempt(attempt)

            # 1. Select target slice from graph
            slice_nodes = self.graph_store.get_partition_nodes(spec)

            # 2. Build harness binary
            target = self.harness_builder.build(slice_nodes)

            # 3. Create verification request
            request = self._create_request(target, spec, budget)

            # 4. Run verification
            result = self.backend.verify(request)
            history.append({
                "attempt": attempt,
                "budget": budget.__dict__,
                "status": result.status,
                "stats": getattr(result, "stats", {}),
            })

            # 5. Evaluate result
            if result.status == "ce_found":
                if self._check_spurious(result, spec):
                    # Spurious CE — refine and retry
                    continue
                return OrchestratorResult(
                    verdict="ce_confirmed",
                    final_result=result,
                    attempts=attempt + 1,
                    history=history,
                )

            if result.status == "no_ce_within_budget":
                if attempt < self.max_refinements - 1:
                    # Retry with larger budget
                    continue
                return OrchestratorResult(
                    verdict="bounded_safe",
                    final_result=result,
                    attempts=attempt + 1,
                    history=history,
                )

            # Error
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

    def _budget_for_attempt(self, attempt: int) -> Budget:
        if attempt < len(self.budgets):
            return self.budgets[attempt]
        # Fallback: double the last preset
        last = self.budgets[-1]
        factor = 2 ** (attempt - len(self.budgets) + 1)
        return Budget(
            timeout_s=last.timeout_s * factor,
            max_states=last.max_states * factor,
            max_path_len=last.max_path_len * factor,
            max_loop_iters=last.max_loop_iters * factor,
        )

    def _create_request(self, target: Any, spec: PropertySpec, budget: Budget) -> Any:
        """Assemble a VerificationRequest dict (JSON-serializable)."""
        return {
            "target": {
                "binary_name": getattr(target, "binary_name", "harness"),
                "entry": getattr(target, "entry", spec.find_value),
            },
            "symbols": getattr(target, "symbols", []),
            "constraints": spec.constraint_templates or [],
            "find": {"kind": spec.find_kind, "value": spec.find_value},
            "avoid": (
                {"kind": spec.avoid_kind, "value": spec.avoid_value}
                if spec.avoid_kind
                else None
            ),
            "budget": budget.__dict__,
            "metadata": {"property": spec.description, "target_hint": spec.target_hint},
        }

    def _check_spurious(self, result: Any, spec: PropertySpec) -> bool:
        """Check if a counterexample is spurious.

        Placeholder: always returns False (CE accepted).
        Replace with cheaper-abstraction cross-check in production.
        """
        # TODO: Implement spurious CE detection
        # Options:
        #   1. Re-run with tighter constraints
        #   2. Check CE against type-level / CFG-level abstraction
        #   3. Use Z3 directly on the path constraints
        return False
