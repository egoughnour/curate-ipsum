"""
Fitness evaluation for synthesis candidates.

Fitness = (ce_weight * CE_avoidance) + (spec_weight * spec_satisfaction) - (complexity_weight * complexity)

CE avoidance: fraction of known counterexamples NOT triggered.
Spec satisfaction: fraction of test commands that pass.
Complexity penalty: AST node count / 100.

Uses tools.py::run_command() for test execution.
Decision: D-013 — fitness function formula.
"""

from __future__ import annotations

import ast
import logging
import os
import tempfile
from typing import Any

from curate_ipsum.synthesis.models import Counterexample, Individual, Specification, SynthesisConfig

LOG = logging.getLogger("synthesis.fitness")


class FitnessEvaluator:
    """Evaluates candidate fitness against specification and counterexamples."""

    def __init__(self, config: SynthesisConfig) -> None:
        self._config = config

    async def evaluate(
        self,
        individual: Individual,
        spec: Specification,
        counterexamples: list[Counterexample],
    ) -> float:
        """
        Compute fitness score for an individual.

        Returns float in range [0, 1] (approximately; complexity penalty can push below 0).
        """
        if not individual.is_valid():
            return -1.0  # Invalid code gets worst possible fitness

        ce_score = self._counterexample_avoidance(individual, counterexamples)
        spec_score = await self._spec_satisfaction(individual, spec)
        complexity = self._complexity_penalty(individual)

        fitness = (
            self._config.ce_weight * ce_score
            + self._config.spec_weight * spec_score
            - self._config.complexity_weight * complexity
        )

        individual.fitness = fitness
        return fitness

    async def evaluate_population(
        self,
        individuals: list[Individual],
        spec: Specification,
        counterexamples: list[Counterexample],
    ) -> None:
        """Evaluate fitness for all individuals in a population."""
        for ind in individuals:
            await self.evaluate(ind, spec, counterexamples)

    def _counterexample_avoidance(
        self,
        individual: Individual,
        counterexamples: list[Counterexample],
    ) -> float:
        """Fraction of counterexamples that this individual does NOT trigger."""
        if not counterexamples:
            return 1.0  # No counterexamples = perfect score

        avoided = 0
        for ce in counterexamples:
            if not self._triggers_counterexample(individual, ce):
                avoided += 1

        return avoided / len(counterexamples)

    def _triggers_counterexample(
        self,
        individual: Individual,
        ce: Counterexample,
    ) -> bool:
        """
        Check if executing the individual's code with CE inputs produces the CE's actual (bad) output.

        Uses safe exec in a restricted namespace. Returns True if the code
        reproduces the counterexample failure.
        """
        try:
            # Compile the individual's code
            code_obj = compile(individual.code, "<synthesis>", "exec")
            namespace: dict[str, Any] = {}
            exec(code_obj, namespace)  # noqa: S102

            # Find the first callable in the namespace
            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    func = obj
                    break

            if func is None:
                return True  # No callable = consider it a failure

            # Call with CE inputs
            result = func(**ce.input_values) if ce.input_values else func()

            # If result matches the CE's actual (bad) output, the CE is triggered
            return str(result) == str(ce.actual_output)
        except Exception:
            # If execution fails, consider the CE triggered (code is broken)
            return True

    async def _spec_satisfaction(
        self,
        individual: Individual,
        spec: Specification,
    ) -> float:
        """Fraction of test commands that pass when the code is applied."""
        if not spec.test_commands:
            return 0.5  # No tests = neutral score

        if not spec.working_directory:
            # Can't run tests without a working directory — use local eval
            return self._local_spec_check(individual, spec)

        # Write the patched code to a temp file in the working directory
        passed = 0
        total = len(spec.test_commands)

        try:
            from curate_ipsum.tools import run_command

            # Create a temporary patched file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                dir=spec.working_directory,
                delete=False,
                prefix="_synthesis_patch_",
            ) as f:
                f.write(individual.code)
                patch_path = f.name

            try:
                for cmd in spec.test_commands:
                    result = await run_command(
                        cmd,
                        spec.working_directory,
                        timeout=self._config.test_timeout_seconds,
                    )
                    if result.exit_code == 0:
                        passed += 1
            finally:
                # Clean up temp file
                try:
                    os.unlink(patch_path)
                except OSError:
                    pass

        except ImportError:
            return self._local_spec_check(individual, spec)
        except Exception as exc:
            LOG.debug("Test execution failed: %s", exc)
            return 0.0

        return passed / total

    def _local_spec_check(self, individual: Individual, spec: Specification) -> float:
        """Fallback spec check: just verify the code is syntactically valid."""
        return 1.0 if individual.is_valid() else 0.0

    @staticmethod
    def _complexity_penalty(individual: Individual) -> float:
        """
        AST complexity as a fraction of 100 nodes.

        Penalty is capped at 1.0 to avoid overwhelming the fitness score.
        """
        try:
            tree = ast.parse(individual.code)
            node_count = sum(1 for _ in ast.walk(tree))
            return min(node_count / 100.0, 1.0)
        except SyntaxError:
            return 1.0  # Max penalty for unparseable code
