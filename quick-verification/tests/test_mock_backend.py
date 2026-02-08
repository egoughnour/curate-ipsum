"""Tests for MockBackend."""
from quickverify.verification.backends.mock import MockBackend
from quickverify.verification.types import (
    Budget,
    Predicate,
    SymbolSpec,
    VerificationRequest,
    VerificationTarget,
)


def _make_req():
    return VerificationRequest(
        target=VerificationTarget(binary_name="h", entry="f"),
        symbols=[SymbolSpec(name="x", bits=32, kind="int")],
        constraints=["x>=0"],
        find=Predicate(kind="addr_reached", value="violation"),
    )


class TestMockBackend:
    def test_no_ce_mode(self):
        b = MockBackend(mode="no_ce")
        r = b.verify(_make_req())
        assert r.status == "no_ce_within_budget"
        assert r.counterexample is None

    def test_ce_mode(self):
        b = MockBackend(mode="ce")
        r = b.verify(_make_req())
        assert r.status == "ce_found"
        assert r.counterexample is not None
        assert "x" in r.counterexample.model

    def test_error_mode(self):
        b = MockBackend(mode="error")
        r = b.verify(_make_req())
        assert r.status == "error"
        assert r.logs == "mock error"

    def test_call_log(self):
        b = MockBackend(mode="no_ce")
        req = _make_req()
        b.verify(req)
        b.verify(req)
        assert len(b.call_log) == 2

    def test_supports(self):
        b = MockBackend()
        s = b.supports()
        assert s["input"] == "mock"
