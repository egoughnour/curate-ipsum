"""Tests for the verification orchestrator — CEGAR loop with budget escalation."""

import pytest

from curate_ipsum.verification.backends.mock import MockBackend
from curate_ipsum.verification.orchestrator import (
    DEFAULT_BUDGET_PRESETS,
    OrchestratorResult,
    VerificationOrchestrator,
)
from curate_ipsum.verification.types import (
    Budget,
    SymbolSpec,
    VerificationRequest,
)


def _make_request() -> VerificationRequest:
    return VerificationRequest(
        target_binary="test.bin",
        entry="main",
        symbols=[SymbolSpec(name="x")],
        find_kind="addr_reached",
        find_value="0x401000",
        budget=Budget(timeout_s=10),
    )


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_ce_found_immediately(self):
        """If backend finds CE on first try, orchestrator should return ce_found."""
        backend = MockBackend(mode="ce")
        orch = VerificationOrchestrator(backend=backend)
        result = await orch.run(_make_request())

        assert result.status == "ce_found"
        assert result.counterexample is not None
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_bounded_safe_all_budgets(self):
        """If no CE at any budget level, should return bounded_safe."""
        backend = MockBackend(mode="no_ce")
        orch = VerificationOrchestrator(backend=backend)
        result = await orch.run(_make_request())

        assert result.status == "bounded_safe"
        assert result.iterations == len(DEFAULT_BUDGET_PRESETS)
        assert result.counterexample is None

    @pytest.mark.asyncio
    async def test_error_stops_early(self):
        """If backend errors, orchestrator should stop and return error."""
        backend = MockBackend(mode="error")
        orch = VerificationOrchestrator(backend=backend)
        result = await orch.run(_make_request())

        assert result.status == "error"
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_spurious_checker_causes_escalation(self):
        """If spurious checker rejects CE, orchestrator should escalate."""
        backend = MockBackend(mode="ce")
        call_count = [0]

        def spurious_checker(ce):
            call_count[0] += 1
            # First CE is spurious, second is real
            return call_count[0] <= 1

        orch = VerificationOrchestrator(
            backend=backend,
            spurious_checker=spurious_checker,
        )
        result = await orch.run(_make_request())

        assert result.status == "ce_found"
        assert result.iterations == 2  # First was spurious, second confirmed

    @pytest.mark.asyncio
    async def test_custom_budget_presets(self):
        """Orchestrator should use custom budget presets."""
        custom_budgets = [
            Budget(timeout_s=5, max_states=100, max_path_len=50, max_loop_iters=1),
            Budget(timeout_s=15, max_states=500, max_path_len=150, max_loop_iters=3),
        ]
        backend = MockBackend(mode="no_ce")
        orch = VerificationOrchestrator(backend=backend, budget_presets=custom_budgets)
        result = await orch.run(_make_request())

        assert result.status == "bounded_safe"
        assert result.iterations == 2
        assert len(result.backend_results) == 2

    @pytest.mark.asyncio
    async def test_multi_backend(self):
        """Multi-backend should try backends in order."""
        backends = [
            MockBackend(mode="no_ce"),  # First: no CE
            MockBackend(mode="ce"),  # Second: finds CE
        ]
        orch = VerificationOrchestrator(backend=backends[0])
        result = await orch.run_multi_backend(_make_request(), backends)

        assert result.status == "ce_found"
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_multi_backend_all_safe(self):
        """If all backends return no CE, multi-backend should return bounded_safe."""
        backends = [
            MockBackend(mode="no_ce"),
            MockBackend(mode="no_ce"),
        ]
        orch = VerificationOrchestrator(backend=backends[0])
        result = await orch.run_multi_backend(_make_request(), backends)

        assert result.status == "bounded_safe"

    def test_orchestrator_result_to_dict(self):
        r = OrchestratorResult(
            status="bounded_safe",
            iterations=3,
            total_elapsed_s=5.5,
            logs=["iter 1: no CE", "iter 2: no CE"],
        )
        d = r.to_dict()
        assert d["status"] == "bounded_safe"
        assert d["iterations"] == 3
        assert len(d["logs"]) == 2
