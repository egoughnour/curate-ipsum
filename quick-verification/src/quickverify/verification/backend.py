"""IVerificationBackend — abstract base class for all verification engines."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .types import VerificationRequest, VerificationResult


class IVerificationBackend(ABC):
    """Abstract verification backend interface.

    Implementations:
      - AngrBackendDocker  (binary symbolic execution in container)
      - Z3Backend          (pure constraint satisfaction, in-process)
      - MockBackend        (deterministic test doubles)
    """

    @abstractmethod
    def verify(self, req: VerificationRequest) -> VerificationResult:
        """Execute a bounded verification run and return the result."""
        raise NotImplementedError

    @abstractmethod
    def supports(self) -> dict:
        """Report backend capabilities for runtime feature negotiation.

        Returns a dict like:
            {"input": "binary"|"constraints", "constraints": [...], "find": [...], ...}
        """
        raise NotImplementedError
