"""Tests for verification.types — data models and serialization."""

from curate_ipsum.verification.types import (
    Budget,
    Counterexample,
    SymbolSpec,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)


class TestBudget:
    def test_defaults(self):
        b = Budget()
        assert b.timeout_s == 30
        assert b.max_states == 50_000
        assert b.max_path_len == 200
        assert b.max_loop_iters == 5

    def test_escalate(self):
        b = Budget(timeout_s=10, max_states=1000, max_path_len=100, max_loop_iters=2)
        b2 = b.escalate(3.0)
        assert b2.timeout_s == 30
        assert b2.max_states == 3000
        assert b2.max_path_len == 300
        assert b2.max_loop_iters == 6
        # Original unchanged
        assert b.timeout_s == 10

    def test_roundtrip(self):
        b = Budget(timeout_s=60, max_states=100_000, max_path_len=500, max_loop_iters=10)
        d = b.to_dict()
        b2 = Budget.from_dict(d)
        assert b2.timeout_s == b.timeout_s
        assert b2.max_states == b.max_states
        assert b2.max_path_len == b.max_path_len
        assert b2.max_loop_iters == b.max_loop_iters


class TestSymbolSpec:
    def test_int_symbol(self):
        s = SymbolSpec(name="x", kind="int", bits=32)
        d = s.to_dict()
        assert d["name"] == "x"
        assert d["kind"] == "int"
        assert d["bits"] == 32
        assert "length" not in d

    def test_bytes_symbol(self):
        s = SymbolSpec(name="buf", kind="bytes", bits=8, length=16)
        d = s.to_dict()
        assert d["length"] == 16

    def test_roundtrip(self):
        s = SymbolSpec(name="flag", kind="bool", bits=1)
        d = s.to_dict()
        s2 = SymbolSpec.from_dict(d)
        assert s2.name == s.name
        assert s2.kind == s.kind
        assert s2.bits == s.bits


class TestVerificationRequest:
    def test_to_dict(self):
        req = VerificationRequest(
            target_binary="test.bin",
            entry="main",
            symbols=[SymbolSpec(name="x", kind="int", bits=64)],
            constraints=["x >= 0", "x <= 100"],
            find_kind="addr_reached",
            find_value="0x401000",
            budget=Budget(timeout_s=10),
            metadata={"test": True},
        )
        d = req.to_dict()
        assert d["target"]["binary_name"] == "test.bin"
        assert d["target"]["entry"] == "main"
        assert len(d["symbols"]) == 1
        assert d["constraints"] == ["x >= 0", "x <= 100"]
        assert d["find"]["kind"] == "addr_reached"
        assert d["budget"]["timeout_s"] == 10

    def test_json_roundtrip(self):
        req = VerificationRequest(
            target_binary="bin.elf",
            entry="target_fn",
            symbols=[SymbolSpec(name="a", kind="int", bits=32)],
            constraints=["a > 0"],
            find_kind="addr_reached",
            find_value="0x400",
        )
        json_str = req.to_json()
        assert '"binary_name": "bin.elf"' in json_str

        import json

        d = json.loads(json_str)
        req2 = VerificationRequest.from_dict(d)
        assert req2.target_binary == req.target_binary
        assert req2.entry == req.entry
        assert len(req2.symbols) == len(req.symbols)


class TestVerificationResult:
    def test_no_ce(self):
        r = VerificationResult(
            status=VerificationStatus.NO_CE_WITHIN_BUDGET,
            stats={"elapsed_s": 1.5},
        )
        d = r.to_dict()
        assert d["status"] == "no_ce_within_budget"
        assert d["counterexample"] is None

    def test_ce_found(self):
        ce = Counterexample(model={"x": 42}, trace=[{"addr": "0x401"}])
        r = VerificationResult(
            status=VerificationStatus.CE_FOUND,
            counterexample=ce,
            stats={"elapsed_s": 0.3},
        )
        d = r.to_dict()
        assert d["status"] == "ce_found"
        assert d["counterexample"]["model"]["x"] == 42

    def test_roundtrip(self):
        ce = Counterexample(model={"a": 10, "b": 20}, notes={"solver": "z3"})
        r = VerificationResult(
            status=VerificationStatus.CE_FOUND,
            counterexample=ce,
            stats={"elapsed_s": 0.1},
        )
        d = r.to_dict()
        r2 = VerificationResult.from_dict(d)
        assert r2.status == VerificationStatus.CE_FOUND
        assert r2.counterexample.model == {"a": 10, "b": 20}
