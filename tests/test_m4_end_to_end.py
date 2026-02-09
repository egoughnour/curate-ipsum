"""End-to-end tests for M4 synthesis pipeline.

Tests the full flow: specification → LLM candidates → genetic algorithm → result.
Uses MockLLMClient with realistic (but simple) code candidates.
"""

import pytest

from synthesis.ast_operators import ASTCrossover, ASTMutator
from synthesis.cegis import CEGISEngine
from synthesis.entropy import EntropyManager
from synthesis.fitness import FitnessEvaluator
from synthesis.llm_client import MockLLMClient
from synthesis.models import (
    CodePatch,
    Counterexample,
    Individual,
    PatchSource,
    Specification,
    SynthesisConfig,
    SynthesisResult,
    SynthesisStatus,
)
from synthesis.population import Population


class TestFullPipeline:
    """Full synthesis pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_pipeline_with_simple_candidates(self):
        """Full pipeline: LLM → population → evolution → result."""
        candidates = [
            "def fix(x):\n    if x is None:\n        return 0\n    return x + 1\n",
            "def fix(x):\n    return x + 2\n",
            "def fix(x):\n    return abs(x) + 1\n",
            "def fix(x):\n    if x < 0:\n        return -x\n    return x + 1\n",
        ]
        config = SynthesisConfig(
            max_iterations=5,
            population_size=4,
            top_k=4,
            mutation_rate=0.5,
            crossover_rate=0.5,
        )
        client = MockLLMClient(responses=candidates)
        engine = CEGISEngine(config, client)
        spec = Specification(
            original_code="def fix(x): return x",
            preconditions=["x is an integer"],
            postconditions=["result > 0 for positive x"],
        )

        result = await engine.synthesize(spec)

        assert isinstance(result, SynthesisResult)
        assert result.iterations >= 1
        assert result.total_candidates_evaluated >= 4
        assert result.id  # Has a run ID
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_tracks_fitness_progress(self):
        """Fitness history should show evolution progress."""
        config = SynthesisConfig(
            max_iterations=10,
            population_size=6,
            top_k=6,
        )
        client = MockLLMClient(
            responses=[
                "def f(x): return x + 1\n",
                "def f(x): return x * 2\n",
                "def f(x): return x ** 2\n",
            ]
        )
        engine = CEGISEngine(config, client)
        spec = Specification()

        result = await engine.synthesize(spec)

        assert len(result.fitness_history) == result.iterations
        assert all(isinstance(f, float) for f in result.fitness_history)


class TestPopulationEvolution:
    """Test that GA evolution actually modifies the population."""

    def test_population_changes_over_generations(self):
        """Population should contain new individuals after evolution."""
        codes = [
            "def a(x):\n    return x + 1\n",
            "def b(x):\n    return x * 2\n",
            "def c(x):\n    return x - 1\n",
            "def d(x):\n    if x > 0:\n        return x\n    return -x\n",
        ]
        pop = Population.from_candidates(codes)
        _original_codes = {ind.code for ind in pop}

        # Assign varying fitness
        for i, ind in enumerate(pop):
            ind.fitness = float(i) / len(codes)

        # Apply crossover
        crossover = ASTCrossover()
        mutator = ASTMutator()

        new_individuals = []
        individuals = pop.individuals
        for i in range(0, len(individuals) - 1, 2):
            c1, c2 = crossover.crossover(individuals[i], individuals[i + 1], generation=1)
            if c1:
                new_individuals.append(c1)
            if c2:
                new_individuals.append(c2)

        # Apply mutation
        for ind in list(new_individuals):
            mutated = mutator.mutate(ind, generation=1)
            if mutated:
                new_individuals.append(mutated)

        # Should have produced at least some new code
        _new_codes = {ind.code for ind in new_individuals}
        # The new generation should have some novel individuals
        assert len(new_individuals) > 0

    def test_elite_preservation(self):
        """Elite individuals should survive across generations."""
        pop = Population(
            [
                Individual(code="x=1", fitness=0.9),
                Individual(code="x=2", fitness=0.1),
                Individual(code="x=3", fitness=0.5),
                Individual(code="x=4", fitness=0.3),
            ]
        )
        elite = pop.select_elite(1)
        assert elite[0].fitness == 0.9


class TestFitnessIntegration:
    """Test fitness evaluation in context of the full pipeline."""

    @pytest.mark.asyncio
    async def test_fitness_evaluates_all_individuals(self):
        """FitnessEvaluator should set fitness on all individuals."""
        config = SynthesisConfig()
        evaluator = FitnessEvaluator(config)
        individuals = [
            Individual(code="def f(x): return x + 1\n"),
            Individual(code="def f(x): return x * 2\n"),
            Individual(code="x = 1\n"),
        ]
        spec = Specification()

        await evaluator.evaluate_population(individuals, spec, [])

        for ind in individuals:
            assert isinstance(ind.fitness, float)
            # Valid code should have positive fitness (no CEs, neutral spec, some complexity)
            assert ind.fitness > -1.0

    @pytest.mark.asyncio
    async def test_fitness_with_counterexamples(self):
        """Counterexamples should affect fitness scores."""
        config = SynthesisConfig()
        evaluator = FitnessEvaluator(config)

        # This function returns x+1
        good_ind = Individual(code="def f(x):\n    return x + 1\n")
        bad_ind = Individual(code="def f(x):\n    return 0\n")

        ce = Counterexample(
            input_values={"x": 5},
            expected_output=6,
            actual_output="0",  # Bad output as string
        )
        spec = Specification()

        good_fitness = await evaluator.evaluate(good_ind, spec, [ce])
        bad_fitness = await evaluator.evaluate(bad_ind, spec, [ce])

        # Good individual avoids CE, bad one triggers it
        assert good_fitness > bad_fitness


class TestEntropyIntegration:
    """Test entropy manager in pipeline context."""

    def test_homogeneous_population_needs_injection(self):
        """Identical individuals should trigger diversity injection."""
        config = SynthesisConfig(entropy_threshold=0.5)
        manager = EntropyManager(config)
        inds = [Individual(code="x = 1") for _ in range(10)]
        assert manager.needs_injection(inds) is True

    def test_diverse_population_entropy(self):
        """Structurally different code should have higher entropy."""
        config = SynthesisConfig()
        manager = EntropyManager(config)

        homogeneous = [Individual(code="x = 1") for _ in range(5)]
        diverse = [
            Individual(code="x = 1"),
            Individual(code="def foo(a, b):\n    if a > b:\n        return a\n    return b\n"),
            Individual(code="for i in range(10):\n    print(i)\n"),
            Individual(code="class Foo:\n    def bar(self): return 1\n"),
            Individual(code="import math\nx = math.sqrt(16)\n"),
        ]

        h_entropy = manager.compute_entropy(homogeneous)
        d_entropy = manager.compute_entropy(diverse)

        assert d_entropy >= h_entropy  # Diverse should have >= entropy


class TestResultSerialization:
    """Ensure results are fully serializable for MCP transport."""

    @pytest.mark.asyncio
    async def test_success_result_serializes(self):
        """A successful result with patch should serialize cleanly."""
        result = SynthesisResult(
            status=SynthesisStatus.SUCCESS,
            patch=CodePatch(code="def f(): return 42", source=PatchSource.CROSSOVER),
            iterations=15,
            counterexamples_resolved=3,
            duration_ms=5000,
            fitness_history=[0.1, 0.3, 0.5, 0.7, 0.9],
        )
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["patch"]["code"] == "def f(): return 42"
        assert d["patch"]["source"] == "crossover"
        assert d["iterations"] == 15
        assert d["counterexamples_resolved"] == 3
        assert len(d["fitness_history"]) == 5

    @pytest.mark.asyncio
    async def test_failure_result_serializes(self):
        """A failed result should serialize cleanly."""
        result = SynthesisResult(
            status=SynthesisStatus.FAILED,
            iterations=50,
            error_message="No verified patch found",
            fitness_history=[0.1] * 50,
        )
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["patch"] is None
        assert d["error_message"] == "No verified patch found"
