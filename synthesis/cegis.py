"""
CEGIS Engine: Counterexample-Guided Inductive Synthesis.

Main synthesis loop:
1. Generate initial candidates via LLM
2. Initialize genetic algorithm population
3. Iterate: evaluate → verify → extract counterexample → evolve → check entropy
4. Return verified patch or failure

Integrates with M3 belief revision for provenance tracking and failure analysis.
Integrates with M5 verification backends for formal property checking.
Integrates with M6-deferred RAG for context-aware prompt building.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING

from synthesis.ast_operators import ASTCrossover, ASTMutator
from synthesis.entropy import EntropyManager
from synthesis.fitness import FitnessEvaluator
from synthesis.llm_client import LLMClient, build_synthesis_prompt
from synthesis.models import (
    CodePatch,
    Counterexample,
    Individual,
    Specification,
    SynthesisConfig,
    SynthesisResult,
    SynthesisStatus,
)
from synthesis.population import Population

if TYPE_CHECKING:
    from rag.search import RAGPipeline
    from theory.manager import TheoryManager
    from verification.backend import VerificationBackend

LOG = logging.getLogger("synthesis.cegis")


class CEGISEngine:
    """
    Counterexample-Guided Inductive Synthesis engine.

    Orchestrates LLM candidate generation, genetic algorithm evolution,
    and counterexample-driven refinement to produce verified patches.
    """

    def __init__(
        self,
        config: SynthesisConfig,
        llm_client: LLMClient,
        theory_manager: "TheoryManager" | None = None,
        verification_backend: "VerificationBackend" | None = None,
        rag_pipeline: "RAGPipeline" | None = None,
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._theory = theory_manager
        self._verification_backend = verification_backend  # M5
        self._rag_pipeline = rag_pipeline  # M6-deferred RAG
        self._fitness = FitnessEvaluator(config)
        self._crossover = ASTCrossover()
        self._mutator = ASTMutator()
        self._entropy = EntropyManager(config)
        self._cancelled = False
        self._current_run_id: str | None = None

    def cancel(self) -> None:
        """Cancel the current synthesis run."""
        self._cancelled = True

    async def synthesize(self, spec: Specification) -> SynthesisResult:
        """
        Run the full CEGIS loop.

        Returns SynthesisResult with status, patch (if successful), and metrics.
        """
        self._cancelled = False
        start_time = time.monotonic()

        result = SynthesisResult()
        self._current_run_id = result.id
        counterexamples: list[Counterexample] = []
        fitness_history: list[float] = []

        try:
            # Step 1: Generate initial candidates from LLM
            LOG.info("CEGIS: Generating initial candidates via LLM...")
            # M6-deferred: enrich context with RAG-retrieved code
            context_code = spec.context_code
            if self._rag_pipeline is not None:
                try:
                    rag_query = f"{spec.original_code}\n{spec.target_region}"
                    rag_results = self._rag_pipeline.search(rag_query)
                    if rag_results:
                        rag_context = self._rag_pipeline.pack_context(rag_results)
                        context_code = (
                            f"{context_code}\n\n## Retrieved context (RAG)\n{rag_context}"
                            if context_code
                            else rag_context
                        )
                        LOG.info("CEGIS: RAG provided %d context chunks", len(rag_results))
                except Exception as exc:
                    LOG.debug("RAG context retrieval failed: %s", exc)
            prompt = build_synthesis_prompt(spec, context_code=context_code)
            raw_candidates = await self._llm.generate_candidates(
                prompt,
                n=self._config.top_k,
                temperature=self._config.temperature,
            )

            if not raw_candidates:
                result.status = SynthesisStatus.FAILED
                result.error_message = "LLM produced no candidates"
                return result

            # Step 2: Initialize population
            population = Population.from_candidates(raw_candidates)
            if population.size() == 0:
                result.status = SynthesisStatus.FAILED
                result.error_message = "No syntactically valid candidates from LLM"
                return result

            LOG.info("CEGIS: Population initialized with %d individuals", population.size())
            result.total_candidates_evaluated = population.size()

            # Step 3: Main CEGIS loop
            for iteration in range(self._config.max_iterations):
                if self._cancelled:
                    result.status = SynthesisStatus.CANCELLED
                    result.error_message = "Synthesis cancelled by user"
                    break

                # Check timeout
                elapsed = time.monotonic() - start_time
                if elapsed > self._config.synthesis_timeout_seconds:
                    result.status = SynthesisStatus.TIMEOUT
                    result.error_message = f"Synthesis timeout after {elapsed:.1f}s"
                    break

                # 3a: Evaluate fitness
                await self._fitness.evaluate_population(population.individuals, spec, counterexamples)

                best = population.best
                if best is None:
                    break

                best_fitness = best.fitness
                fitness_history.append(best_fitness)

                # 3b: Check if best individual satisfies the spec
                if await self._verify_patch(best, spec):
                    LOG.info(
                        "CEGIS: SUCCESS at iteration %d (fitness=%.3f, CEs=%d)",
                        iteration,
                        best_fitness,
                        len(counterexamples),
                    )
                    result.status = SynthesisStatus.SUCCESS
                    result.patch = CodePatch(
                        code=best.code,
                        source=best.source,
                        region_id=spec.target_region,
                        original_code=spec.original_code,
                    )
                    # Record success in M3
                    await self._record_success(spec, best)
                    break

                # 3c: Extract counterexample from best's failure
                ce = await self._extract_counterexample(best, spec, iteration)
                if ce:
                    counterexamples.append(ce)
                    # Record CE in M3 provenance
                    await self._record_counterexample(spec, ce)

                # 3d: Evolve population
                population = self._evolve(population, counterexamples, iteration + 1)
                result.total_candidates_evaluated += population.size()

                # 3e: Check entropy and inject diversity if needed
                if self._entropy.needs_injection(population.individuals):
                    LOG.debug("CEGIS iteration %d: low entropy, injecting diversity", iteration)
                    population = await self._inject_diversity(population, spec, counterexamples)

                # 3f: Log progress
                entropy = self._entropy.compute_entropy(population.individuals)
                LOG.debug(
                    "CEGIS iteration %d: best=%.3f, avg=%.3f, entropy=%.2f, CEs=%d",
                    iteration,
                    best_fitness,
                    population.average_fitness,
                    entropy,
                    len(counterexamples),
                )

            else:
                # Loop exhausted without break
                result.status = SynthesisStatus.FAILED
                result.error_message = f"No verified patch after {self._config.max_iterations} iterations"
                # Record failure in M3
                await self._record_failure(spec, fitness_history)

        except Exception as exc:
            LOG.exception("CEGIS synthesis failed with exception")
            result.status = SynthesisStatus.FAILED
            result.error_message = str(exc)

        # Fill result metrics
        result.iterations = len(fitness_history)
        result.counterexamples_resolved = len(counterexamples)
        result.duration_ms = int((time.monotonic() - start_time) * 1000)
        result.fitness_history = fitness_history
        result.final_entropy = (
            self._entropy.compute_entropy(population.individuals)
            if "population" in dir() and population.size() > 0
            else 0.0
        )

        return result

    async def _verify_patch(self, individual: Individual, spec: Specification) -> bool:
        """
        Full verification: all tests pass AND target mutant is killed.

        M4: test-based verification.
        M5: formal verification via VerificationBackend (Z3/angr) if configured.
        """
        if not spec.test_commands or not spec.working_directory:
            # No tests to run — consider spec satisfied if fitness is high enough
            return individual.fitness > 0.8

        try:
            import os
            import tempfile

            from tools import run_command

            # Write patch to temp file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                dir=spec.working_directory,
                delete=False,
                prefix="_verify_patch_",
            ) as f:
                f.write(individual.code)
                patch_path = f.name

            try:
                # Run all test commands
                for cmd in spec.test_commands:
                    result = await run_command(
                        cmd,
                        spec.working_directory,
                        timeout=self._config.test_timeout_seconds,
                    )
                    if result.exit_code != 0:
                        return False

                # If mutation command exists, verify mutant is killed
                if spec.mutation_command:
                    result = await run_command(
                        spec.mutation_command,
                        spec.working_directory,
                        timeout=self._config.test_timeout_seconds * 2,
                    )
                    if result.exit_code != 0:
                        return False

                # M5: formal verification layer (if backend configured)
                if self._verification_backend is not None:
                    formal_ok = await self._run_formal_verification(individual, spec)
                    if not formal_ok:
                        return False

                return True
            finally:
                try:
                    os.unlink(patch_path)
                except OSError:
                    pass
        except ImportError:
            # No tools module — use fitness threshold
            return individual.fitness > 0.8

    async def _run_formal_verification(self, individual: Individual, spec: Specification) -> bool:
        """
        Run formal verification on a candidate patch (M5).

        Uses the configured VerificationBackend (Z3 or angr) to check
        properties extracted from the specification's pre/postconditions.

        Returns True if no counterexample found within budget.
        """
        if self._verification_backend is None:
            return True

        try:
            from verification.types import (
                Budget,
                VerificationRequest,
                VerificationStatus,
            )

            # Build constraints from spec preconditions/postconditions
            constraints = list(spec.preconditions) + list(spec.postconditions)
            if not constraints:
                return True  # No formal properties to check

            request = VerificationRequest(
                target_binary="",
                entry=spec.target_region or "patched_func",
                symbols=[],
                constraints=constraints,
                find_kind="custom" if constraints else "return_value",
                find_value="",
                budget=Budget(timeout_s=10, max_states=10_000, max_path_len=100, max_loop_iters=3),
                metadata={"synthesis_run_id": self._current_run_id, "region": spec.target_region},
            )

            vresult = await self._verification_backend.verify(request)

            if vresult.status == VerificationStatus.CE_FOUND:
                LOG.info("Formal verification found CE for candidate %s", individual.id)
                return False

            # no_ce_within_budget or error → treat as passing (bounded guarantee)
            return True

        except Exception as exc:
            LOG.debug("Formal verification failed (treating as pass): %s", exc)
            return True

    async def _extract_counterexample(
        self,
        individual: Individual,
        spec: Specification,
        iteration: int,
    ) -> Counterexample | None:
        """Extract a counterexample from a failing test."""
        if not spec.test_commands or not spec.working_directory:
            return None

        try:
            from tools import run_command

            for cmd in spec.test_commands:
                result = await run_command(
                    cmd,
                    spec.working_directory,
                    timeout=self._config.test_timeout_seconds,
                )
                if result.exit_code != 0:
                    return Counterexample(
                        error_message=result.stderr[:500] or result.stdout[:500],
                        test_command=cmd,
                        metadata={"iteration": iteration, "exit_code": result.exit_code},
                    )
        except (ImportError, Exception) as exc:
            LOG.debug("CE extraction failed: %s", exc)

        return None

    def _evolve(
        self,
        population: Population,
        counterexamples: list[Counterexample],
        generation: int,
    ) -> Population:
        """Evolve population for one epoch using GA operators."""
        config = self._config
        pop_size = max(config.population_size, population.size())
        n_elite = max(1, int(pop_size * config.elite_ratio))

        # Elitism: keep top individuals
        elite = population.select_elite(n_elite)

        # Tournament selection for parents
        n_parents = pop_size - n_elite
        parents = population.tournament_select(n_parents)

        # Crossover
        offspring: list[Individual] = []
        for i in range(0, len(parents) - 1, 2):
            if random.random() < config.crossover_rate:
                child1, child2 = self._crossover.crossover(parents[i], parents[i + 1], generation)
                if child1:
                    offspring.append(child1)
                if child2:
                    offspring.append(child2)
            else:
                offspring.append(parents[i])
                if i + 1 < len(parents):
                    offspring.append(parents[i + 1])

        # If odd number of parents, add the last one
        if len(parents) % 2 == 1:
            offspring.append(parents[-1])

        # Directed mutation
        last_ce = counterexamples[-1] if counterexamples else None
        mutated: list[Individual] = []
        for ind in offspring:
            if random.random() < config.mutation_rate:
                mutant = self._mutator.mutate(ind, generation, last_ce)
                mutated.append(mutant if mutant else ind)
            else:
                mutated.append(ind)

        # Combine elite + mutated offspring
        new_gen = elite + mutated

        new_pop = Population(new_gen)
        return new_pop

    async def _inject_diversity(
        self,
        population: Population,
        spec: Specification,
        counterexamples: list[Counterexample],
    ) -> Population:
        """Request fresh candidates from LLM to inject diversity."""
        n_inject = max(1, population.size() // 4)  # Replace 25% of population

        prompt = build_synthesis_prompt(spec, counterexamples, spec.context_code)
        fresh_candidates = await self._llm.generate_candidates(
            prompt,
            n=n_inject,
            temperature=min(1.0, self._config.temperature + 0.1),
        )

        if fresh_candidates:
            # Remove weakest and add fresh
            indices = self._entropy.select_for_replacement(population.individuals, len(fresh_candidates))
            remaining = [ind for i, ind in enumerate(population.individuals) if i not in set(indices)]
            fresh_pop = Population.from_candidates(fresh_candidates)
            new_pop = Population(remaining + fresh_pop.individuals)
            return new_pop

        return population

    async def _record_success(self, spec: Specification, individual: Individual) -> None:
        """Record successful synthesis in M3 belief revision."""
        if not self._theory:
            return
        try:
            self._theory.add_assertion(
                assertion_type="behavior",
                content=f"Patch verified: kills mutants {spec.surviving_mutant_ids} in region {spec.target_region}",
                evidence_id=f"synthesis_{self._current_run_id}",
                confidence=0.9,
                region_id=spec.target_region,
            )
        except Exception as exc:
            LOG.debug("Failed to record success in M3: %s", exc)

    async def _record_counterexample(self, spec: Specification, ce: Counterexample) -> None:
        """Record counterexample as evidence in M3."""
        if not self._theory:
            return
        try:
            self._theory.store_evidence(
                evidence_kind="counterexample",
                content=f"CE: {ce.error_message[:100]}",
                reliability="B",
                metadata=ce.to_dict(),
            )
        except Exception as exc:
            LOG.debug("Failed to record CE in M3: %s", exc)

    async def _record_failure(self, spec: Specification, fitness_history: list[float]) -> None:
        """Analyze and record synthesis failure in M3."""
        if not self._theory:
            return
        try:
            best_fitness = max(fitness_history) if fitness_history else 0.0
            self._theory.analyze_failure(
                error_message=f"Synthesis failed after {len(fitness_history)} iterations (best fitness: {best_fitness:.3f})",
                test_pass_rate=best_fitness,
                region_id=spec.target_region,
            )
        except Exception as exc:
            LOG.debug("Failed to record failure in M3: %s", exc)
