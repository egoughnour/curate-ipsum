"""
Abstract LLM client for code synthesis.

Defines the LLMClient ABC and MockLLMClient for testing.
Cloud and local backends are in separate modules (cloud_llm.py, local_llm.py).

Design decision: mirrors D-001's dual extractor pattern — abstract base class
with multiple concrete backends selectable at runtime.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional

from synthesis.models import Counterexample, Specification

LOG = logging.getLogger("synthesis.llm_client")


class LLMClient(abc.ABC):
    """Abstract base class for LLM code generation backends."""

    @abc.abstractmethod
    async def generate_candidates(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.8,
    ) -> List[str]:
        """
        Generate n code candidate strings from the LLM.

        Returns raw code strings (not parsed). Caller is responsible for
        syntactic validation.
        """
        ...

    async def close(self) -> None:
        """Clean up resources (e.g., HTTP clients). Override if needed."""
        pass


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.

    Returns canned responses or generates simple variants of a template.
    """

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self._responses = responses or []
        self._call_count = 0

    async def generate_candidates(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.8,
    ) -> List[str]:
        self._call_count += 1
        if self._responses:
            # Return up to n responses, cycling if needed
            result = []
            for i in range(n):
                result.append(self._responses[i % len(self._responses)])
            return result
        # Default: return simple placeholder functions
        return [
            f"def patched_func(x):\n    return x + {i}\n"
            for i in range(n)
        ]

    @property
    def call_count(self) -> int:
        return self._call_count


def build_synthesis_prompt(
    spec: Specification,
    counterexamples: Optional[List[Counterexample]] = None,
    context_code: str = "",
) -> str:
    """
    Build an LLM prompt for code synthesis.

    Includes:
    - The original code being replaced
    - Test requirements
    - Surviving mutant information
    - Counterexample history (what previous attempts failed on)
    - Preconditions and postconditions from M3 assertions
    """
    parts: List[str] = []

    parts.append("Generate a Python function that satisfies the following requirements.\n")

    if spec.original_code:
        parts.append(f"## Original Code\n```python\n{spec.original_code}\n```\n")

    if context_code:
        parts.append(f"## Context\n```python\n{context_code}\n```\n")

    if spec.preconditions:
        parts.append("## Preconditions")
        for pre in spec.preconditions:
            parts.append(f"- {pre}")
        parts.append("")

    if spec.postconditions:
        parts.append("## Postconditions")
        for post in spec.postconditions:
            parts.append(f"- {post}")
        parts.append("")

    if spec.surviving_mutant_ids:
        parts.append(f"## Target: Kill surviving mutants")
        parts.append(f"Mutant IDs: {', '.join(spec.surviving_mutant_ids)}")
        parts.append("The patch must cause these mutants to be detected (killed) by the test suite.\n")

    if spec.test_commands:
        parts.append("## Tests that must pass")
        for cmd in spec.test_commands:
            parts.append(f"- `{cmd}`")
        parts.append("")

    if counterexamples:
        parts.append("## Previous attempts failed on these counterexamples")
        for ce in counterexamples[-5:]:  # Show last 5 CEs to avoid prompt bloat
            parts.append(f"- Input: {ce.input_values}, Expected: {ce.expected_output}, "
                         f"Got: {ce.actual_output}")
            if ce.error_message:
                parts.append(f"  Error: {ce.error_message}")
        parts.append("")

    parts.append("Return ONLY the Python code, no explanations or markdown fences.")

    return "\n".join(parts)
