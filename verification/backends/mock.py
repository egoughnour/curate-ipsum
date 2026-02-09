"""
Mock verification backend for testing.

Mirrors the angr_adapter_baseline mock with additional call logging.
Three modes: "ce" (always finds counterexample), "no_ce" (always safe),
"error" (always errors).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from verification.backend import VerificationBackend
from verification.types import (
    Counterexample,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)

LOG = logging.getLogger("verification.backends.mock")


class MockBackend(VerificationBackend):
    """
    Mock verification backend for testing and CI.

    Modes:
        "ce"    - Always returns a counterexample
        "no_ce" - Always returns no_ce_within_budget
        "error" - Always returns an error
    """

    def __init__(self, mode: str = "no_ce", **kwargs: Any) -> None:
        self.mode = mode
        self.call_log: List[VerificationRequest] = []

    def supports(self) -> Dict[str, Any]:
        return {
            "input": "mock",
            "constraints": ["any"],
            "find": ["any"],
            "avoid": ["any"],
        }

    async def verify(self, request: VerificationRequest) -> VerificationResult:
        t0 = time.monotonic()
        self.call_log.append(request)

        if self.mode == "ce":
            ce = Counterexample(
                model={s.name: 0 for s in request.symbols},
                trace=[{"addr": "0x0"}],
                path_constraints=[],
                notes={"mock": True},
            )
            return VerificationResult(
                status=VerificationStatus.CE_FOUND,
                counterexample=ce,
                stats={"elapsed_s": time.monotonic() - t0, "mock": True},
            )

        if self.mode == "error":
            return VerificationResult(
                status=VerificationStatus.ERROR,
                stats={"elapsed_s": time.monotonic() - t0, "mock": True},
                logs="mock error",
            )

        return VerificationResult(
            status=VerificationStatus.NO_CE_WITHIN_BUDGET,
            stats={"elapsed_s": time.monotonic() - t0, "mock": True},
        )
