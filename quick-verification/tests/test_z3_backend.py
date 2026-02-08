"""Tests for Z3Backend — pure constraint satisfaction."""
import pytest


def _have_z3():
    try:
        import z3
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _have_z3(), reason="z3-solver not installed")


from quickverify.verification.backends.z3_backend import Z3Backend
from quickverify.verification.types import (
    Budget,
    Predicate,
    SymbolSpec,
    VerificationRequest,
    VerificationTarget,
)


class TestZ3Backend:
    def setup_method(self):
        self.backend = Z3Backend()

    def test_supports(self):
        s = self.backend.supports()
        assert s["input"] == "constraints"
        assert "return_value" in s["find"]

    def test_satisfiable_constraint(self):
        """x + y == 1337 with 0<=x<=2000, 0<=y<=2000 should be SAT."""
        req = VerificationRequest(
            target=VerificationTarget(binary_name="n/a", entry="n/a"),
            symbols=[
                SymbolSpec(name="x", bits=32, kind="int"),
                SymbolSpec(name="y", bits=32, kind="int"),
            ],
            constraints=["x>=0", "x<=2000", "y>=0", "y<=2000"],
            find=Predicate(kind="custom", value="x==1337"),
            budget=Budget(timeout_s=5),
        )
        result = self.backend.verify(req)
        # Since we're asking Z3 to find x==1337 with x in [0,2000], it should find it
        assert result.status == "ce_found"
        assert result.counterexample is not None
        assert result.counterexample.model["x"] == 1337

    def test_unsatisfiable_constraint(self):
        """x > 100 AND x < 50 should be UNSAT."""
        req = VerificationRequest(
            target=VerificationTarget(binary_name="n/a", entry="n/a"),
            symbols=[SymbolSpec(name="x", bits=32, kind="int")],
            constraints=["x>100", "x<50"],
            find=Predicate(kind="custom", value="x==75"),
            budget=Budget(timeout_s=5),
        )
        result = self.backend.verify(req)
        assert result.status == "no_ce_within_budget"

    def test_addr_reached_not_supported(self):
        """Z3 backend should error on addr_reached predicates."""
        req = VerificationRequest(
            target=VerificationTarget(binary_name="n/a", entry="n/a"),
            symbols=[SymbolSpec(name="x", bits=32, kind="int")],
            constraints=[],
            find=Predicate(kind="addr_reached", value="0x401000"),
            budget=Budget(timeout_s=5),
        )
        result = self.backend.verify(req)
        assert result.status == "error"
        assert "addr_reached" in (result.logs or "")

    def test_bool_symbol(self):
        """Boolean symbol (1-bit) should work."""
        req = VerificationRequest(
            target=VerificationTarget(binary_name="n/a", entry="n/a"),
            symbols=[SymbolSpec(name="flag", bits=1, kind="bool")],
            constraints=[],
            find=Predicate(kind="custom", value="flag==1"),
            budget=Budget(timeout_s=5),
        )
        result = self.backend.verify(req)
        assert result.status == "ce_found"
        assert result.counterexample.model["flag"] == 1
