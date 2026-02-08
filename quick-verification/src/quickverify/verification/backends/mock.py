"""MockBackend — deterministic test double for IVerificationBackend."""
from __future__ import annotations

from ..backend import IVerificationBackend
from ..types import Counterexample, VerificationRequest, VerificationResult


class MockBackend(IVerificationBackend):
    """Returns canned results for testing orchestrator/pipeline logic."""

    def __init__(self, mode: str = "no_ce"):
        """mode: 'ce' | 'no_ce' | 'error'"""
        self.mode = mode
        self.call_log: list = []

    def supports(self) -> dict:
        return {"input": "mock", "constraints": ["any"], "find": ["any"], "avoid": ["any"]}

    def verify(self, req: VerificationRequest) -> VerificationResult:
        self.call_log.append(req)
        if self.mode == "ce":
            ce = Counterexample(
                model={s.name: 0 for s in req.symbols},
                trace=[{"addr": "0x0"}],
                path_constraints=[],
                notes={"mock": True},
            )
            return VerificationResult(status="ce_found", counterexample=ce, stats={"mock": True})
        if self.mode == "error":
            return VerificationResult(status="error", stats={"mock": True}, logs="mock error")
        return VerificationResult(status="no_ce_within_budget", stats={"mock": True})
