"""Tests for verification data model (types.py)."""
import json
import pytest

from quickverify.verification.types import (
    Budget,
    Counterexample,
    Predicate,
    SymbolSpec,
    VerificationRequest,
    VerificationResult,
    VerificationTarget,
)


class TestBudget:
    def test_defaults(self):
        b = Budget()
        assert b.timeout_s == 10
        assert b.max_states == 256
        assert b.max_path_len == 200
        assert b.max_loop_iters == 8

    def test_escalate(self):
        b = Budget(timeout_s=10, max_states=256, max_path_len=200, max_loop_iters=8)
        b2 = b.escalate(2)
        assert b2.timeout_s == 20
        assert b2.max_states == 512
        assert b2.max_path_len == 400
        assert b2.max_loop_iters == 16

    def test_frozen(self):
        b = Budget()
        with pytest.raises(AttributeError):
            b.timeout_s = 99


class TestVerificationRequest:
    def test_to_json_roundtrip(self, sample_request):
        j = sample_request.to_json()
        assert j["target"]["binary_name"] == "harness"
        assert j["target"]["entry"] == "target_fn"
        assert len(j["symbols"]) == 2
        assert j["constraints"] == ["x>=0", "x<=2000", "y>=0", "y<=2000"]
        assert j["find"]["kind"] == "addr_reached"
        assert j["budget"]["timeout_s"] == 10

        # Roundtrip
        req2 = VerificationRequest.from_json(j)
        assert req2.target.binary_name == sample_request.target.binary_name
        assert req2.symbols == sample_request.symbols
        assert req2.constraints == sample_request.constraints

    def test_to_json_no_avoid(self):
        req = VerificationRequest(
            target=VerificationTarget(binary_name="h", entry="f"),
            symbols=[],
            constraints=[],
            find=Predicate(kind="addr_reached", value="violation"),
        )
        j = req.to_json()
        assert j["avoid"] is None


class TestVerificationResult:
    def test_ce_found_roundtrip(self):
        ce = Counterexample(
            model={"x": 337, "y": 1000},
            trace=[{"addr": "0x401000"}],
            path_constraints=["x + y == 1337"],
            notes={"backend": "angr"},
        )
        r = VerificationResult(status="ce_found", counterexample=ce, stats={"elapsed_s": 1.5})
        j = r.to_json()
        assert j["status"] == "ce_found"
        assert j["counterexample"]["model"]["x"] == 337

        r2 = VerificationResult.from_json(j)
        assert r2.status == "ce_found"
        assert r2.counterexample.model["x"] == 337

    def test_no_ce(self):
        r = VerificationResult(status="no_ce_within_budget", stats={"elapsed_s": 10.0})
        j = r.to_json()
        assert j["counterexample"] is None
        r2 = VerificationResult.from_json(j)
        assert r2.status == "no_ce_within_budget"
        assert r2.counterexample is None
