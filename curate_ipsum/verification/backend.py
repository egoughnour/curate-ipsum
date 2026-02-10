"""
Abstract verification backend interface.

Mirrors D-001/D-012/D-014 pattern: abstract base class with factory function
for runtime backend selection.

Decision: D-016
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from verification.types import VerificationRequest, VerificationResult

LOG = logging.getLogger("verification.backend")


class VerificationBackend(ABC):
    """
    Abstract interface for verification backends.

    Implementations must handle:
    - Accepting a VerificationRequest
    - Running verification within budget constraints
    - Returning a VerificationResult

    Follows the same ABC + factory pattern as GraphStore (D-014)
    and LLMClient (D-012).
    """

    @abstractmethod
    async def verify(self, request: VerificationRequest) -> VerificationResult:
        """
        Run verification on the given request.

        Returns VerificationResult with status, optional counterexample,
        and execution statistics.
        """
        ...

    @abstractmethod
    def supports(self) -> dict[str, Any]:
        """
        Declare what this backend supports.

        Returns a dict describing supported input types, constraint kinds,
        find/avoid predicates, etc. Used by the orchestrator to route
        requests to capable backends.

        Example::

            {"input": "binary", "constraints": ["comparison"], "find": ["addr_reached"], "avoid": ["addr_avoided"]}
        """
        ...

    async def close(self) -> None:
        """Release backend resources. Override if needed."""
        pass


def build_verification_backend(backend: str, **kwargs: Any) -> VerificationBackend:
    """
    Factory: create a VerificationBackend of the requested type.

    Args:
        backend: "z3", "angr", or "mock"
        **kwargs: Backend-specific configuration

    Returns:
        VerificationBackend instance

    Raises:
        ValueError: Unknown backend
        ImportError: Backend dependencies not installed
    """
    if backend == "z3":
        from verification.backends.z3_backend import Z3Backend

        return Z3Backend(**kwargs)

    elif backend == "angr":
        from verification.backends.angr_docker import AngrDockerBackend

        return AngrDockerBackend(**kwargs)

    elif backend == "mock":
        from verification.backends.mock import MockBackend

        return MockBackend(**kwargs)

    else:
        raise ValueError(f"Unknown verification backend: {backend!r}. Supported: 'z3', 'angr', 'mock'")
