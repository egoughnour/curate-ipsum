"""
Verification backends for formal property checking (M5).

Provides abstract verification interface with Z3 (constraint solving),
angr (Docker-based symbolic execution), and mock backends.

Decision: D-016
"""

from __future__ import annotations

from curate_ipsum.verification.types import (
    Budget,
    Counterexample,
    SymbolSpec,
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    "Budget",
    "Counterexample",
    "SymbolSpec",
    "VerificationRequest",
    "VerificationResult",
    "VerificationStatus",
]
