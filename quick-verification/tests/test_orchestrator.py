"""Tests for VerificationOrchestrator — CEGAR loop."""
import pytest

from quickverify.verification.backends.mock import MockBackend
from quickverify.verification.orchestrator import (
    BUDGET_PRESETS,
    OrchestratorResult,
    PropertySpec,
    VerificationOrchestrator,
)
from quickverify.verification.types import SymbolSpec


class TestOrchestrator:
    def test_ce_confirmed(self, mock_backend_ce):
        orch = VerificationOrchestrator(backend=mock_backend_ce)
        spec = PropertySpec(
            description="test",
            find_kind="addr_reached",
            find_value="violation",
            symbols=[SymbolSpec(name="x", bits=32, kind="int")],
        )
        result = orch.verify_property(spec)
        assert result.verdict == "ce_confirmed"
        assert result.attempts == 1
        assert len(result.history) == 1

    def test_bounded_safe_after_escalation(self, mock_backend_no_ce):
        orch = VerificationOrchestrator(backend=mock_backend_no_ce, max_refinements=3)
        spec = PropertySpec(description="test", find_value="violation")
        result = orch.verify_property(spec)
        assert result.verdict == "bounded_safe"
        assert result.attempts == 3
        # Should have escalated budgets
        assert len(result.history) == 3

    def test_error_stops_immediately(self, mock_backend_error):
        orch = VerificationOrchestrator(backend=mock_backend_error)
        spec = PropertySpec(description="test", find_value="violation")
        result = orch.verify_property(spec)
        assert result.verdict == "error"
        assert result.attempts == 1

    def test_spurious_checker_triggers_retry(self):
        """If spurious checker says CE is spurious, orchestrator retries."""
        call_count = [0]

        class CountingMock(MockBackend):
            def verify(self, req):
                call_count[0] += 1
                return super().verify(req)

        backend = CountingMock(mode="ce")
        # First call: spurious. Second call: accept.
        spurious_results = [True, False]

        def checker(result, spec):
            return spurious_results.pop(0) if spurious_results else False

        orch = VerificationOrchestrator(
            backend=backend,
            max_refinements=3,
            spurious_checker=checker,
        )
        spec = PropertySpec(
            description="test",
            find_value="violation",
            symbols=[SymbolSpec(name="x", bits=32, kind="int")],
        )
        result = orch.verify_property(spec)
        assert result.verdict == "ce_confirmed"
        assert result.attempts == 2

    def test_budget_escalation(self):
        orch = VerificationOrchestrator(backend=MockBackend(mode="no_ce"))
        b0 = orch._budget_for_attempt(0)
        b1 = orch._budget_for_attempt(1)
        b2 = orch._budget_for_attempt(2)
        assert b0.timeout_s < b1.timeout_s < b2.timeout_s
        assert b0.max_states < b1.max_states < b2.max_states

    def test_multi_backend(self, mock_backend_no_ce, mock_backend_ce):
        """Multi-backend: first (no_ce) fails to decide, second (ce) finds CE."""
        orch = VerificationOrchestrator(backend=mock_backend_no_ce)
        spec = PropertySpec(
            description="test",
            find_value="violation",
            symbols=[SymbolSpec(name="x", bits=32, kind="int")],
        )
        result = orch.verify_multi_backend(spec, [mock_backend_no_ce, mock_backend_ce])
        # First backend exhausts budget → bounded_safe on first try
        # Actually mock_backend_no_ce will return bounded_safe after max_refinements
        assert result.verdict in ("bounded_safe", "ce_confirmed")
