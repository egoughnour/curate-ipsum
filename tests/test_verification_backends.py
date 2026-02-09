"""Tests for verification backends — mock, Z3, and factory."""

import pytest
from verification.backend import build_verification_backend, VerificationBackend
from verification.backends.mock import MockBackend
from verification.types import (
    Budget,
    SymbolSpec,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)


def _make_request(**kwargs) -> VerificationRequest:
    defaults = dict(
        target_binary="test.bin",
        entry="main",
        symbols=[SymbolSpec(name="x", kind="int", bits=64)],
        constraints=[],
        find_kind="addr_reached",
        find_value="0x401000",
        budget=Budget(timeout_s=5),
    )
    defaults.update(kwargs)
    return VerificationRequest(**defaults)


class TestMockBackend:
    @pytest.mark.asyncio
    async def test_no_ce_mode(self):
        backend = MockBackend(mode="no_ce")
        req = _make_request()
        result = await backend.verify(req)
        assert result.status == VerificationStatus.NO_CE_WITHIN_BUDGET
        assert result.counterexample is None
        assert len(backend.call_log) == 1

    @pytest.mark.asyncio
    async def test_ce_mode(self):
        backend = MockBackend(mode="ce")
        req = _make_request()
        result = await backend.verify(req)
        assert result.status == VerificationStatus.CE_FOUND
        assert result.counterexample is not None
        assert "x" in result.counterexample.model

    @pytest.mark.asyncio
    async def test_error_mode(self):
        backend = MockBackend(mode="error")
        req = _make_request()
        result = await backend.verify(req)
        assert result.status == VerificationStatus.ERROR
        assert result.logs == "mock error"

    def test_supports(self):
        backend = MockBackend()
        s = backend.supports()
        assert s["input"] == "mock"
        assert "any" in s["constraints"]


class TestFactory:
    def test_build_mock(self):
        backend = build_verification_backend("mock")
        assert isinstance(backend, MockBackend)

    def test_build_mock_with_mode(self):
        backend = build_verification_backend("mock", mode="ce")
        assert isinstance(backend, MockBackend)
        assert backend.mode == "ce"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown verification backend"):
            build_verification_backend("nonexistent")


class TestZ3Backend:
    """Tests for Z3 backend — requires z3-solver to be installed."""

    @pytest.mark.asyncio
    async def test_addr_reached_rejected(self):
        """Z3 doesn't support addr_reached — should return error."""
        try:
            from verification.backends.z3_backend import Z3Backend
        except ImportError:
            pytest.skip("z3-solver not installed")

        backend = Z3Backend()
        req = _make_request(find_kind="addr_reached")
        result = await backend.verify(req)
        assert result.status == VerificationStatus.ERROR
        assert "addr_reached" in (result.logs or "")

    @pytest.mark.asyncio
    async def test_satisfiable_constraint(self):
        """Z3 should find a CE for satisfiable constraints."""
        try:
            from verification.backends.z3_backend import Z3Backend
            import z3  # noqa: F401
        except ImportError:
            pytest.skip("z3-solver not installed")

        backend = Z3Backend()
        req = _make_request(
            find_kind="custom",
            find_value="",
            constraints=["x >= 10", "x <= 20"],
        )
        result = await backend.verify(req)
        assert result.status == VerificationStatus.CE_FOUND
        assert result.counterexample is not None
        x_val = result.counterexample.model.get("x")
        assert x_val is not None
        assert 10 <= x_val <= 20

    @pytest.mark.asyncio
    async def test_unsatisfiable_constraint(self):
        """Z3 should report no CE for contradictory constraints."""
        try:
            from verification.backends.z3_backend import Z3Backend
            import z3  # noqa: F401
        except ImportError:
            pytest.skip("z3-solver not installed")

        backend = Z3Backend()
        req = _make_request(
            find_kind="return_value",
            find_value="",
            constraints=["x > 100", "x < 50"],
        )
        result = await backend.verify(req)
        assert result.status == VerificationStatus.NO_CE_WITHIN_BUDGET

    def test_z3_supports(self):
        try:
            from verification.backends.z3_backend import Z3Backend
        except ImportError:
            pytest.skip("z3-solver not installed")

        backend = Z3Backend()
        s = backend.supports()
        assert s["input"] == "constraints"
        assert "addr_reached" not in s["find"]
