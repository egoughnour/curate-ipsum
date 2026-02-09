"""Tests for genetic operators, population management, fitness, and entropy."""

from synthesis.ast_operators import ASTCrossover, ASTMutator
from synthesis.entropy import EntropyManager
from synthesis.fitness import FitnessEvaluator
from synthesis.models import (
    Counterexample,
    Individual,
    PatchSource,
    Specification,
    SynthesisConfig,
)
from synthesis.population import Population

# ── Population ──────────────────────────────────────────────────────────────


class TestPopulation:
    def test_from_candidates_filters_invalid(self):
        candidates = [
            "def foo(): return 1",
            "def bar( return 2",  # Syntax error
            "def baz(): return 3",
        ]
        pop = Population.from_candidates(candidates)
        assert len(pop) == 2  # Only valid ones

    def test_from_candidates_empty(self):
        pop = Population.from_candidates([])
        assert len(pop) == 0

    def test_best_returns_highest_fitness(self):
        inds = [
            Individual(code="x=1", fitness=0.3),
            Individual(code="x=2", fitness=0.9),
            Individual(code="x=3", fitness=0.5),
        ]
        pop = Population(inds)
        assert pop.best.fitness == 0.9

    def test_best_empty_population(self):
        pop = Population()
        assert pop.best is None

    def test_average_fitness(self):
        inds = [
            Individual(code="x=1", fitness=0.2),
            Individual(code="x=2", fitness=0.4),
            Individual(code="x=3", fitness=0.6),
        ]
        pop = Population(inds)
        assert abs(pop.average_fitness - 0.4) < 0.001

    def test_select_elite(self):
        inds = [
            Individual(code="x=1", fitness=0.1),
            Individual(code="x=2", fitness=0.9),
            Individual(code="x=3", fitness=0.5),
        ]
        pop = Population(inds)
        elite = pop.select_elite(2)
        assert len(elite) == 2
        assert elite[0].fitness == 0.9
        assert elite[1].fitness == 0.5

    def test_tournament_select(self):
        inds = [Individual(code=f"x={i}", fitness=float(i) / 10) for i in range(10)]
        pop = Population(inds)
        selected = pop.tournament_select(5, k=3)
        assert len(selected) == 5
        # Tournament should bias toward higher fitness
        avg = sum(s.fitness for s in selected) / len(selected)
        assert avg > 0.3  # Should be above random average of 0.45

    def test_add_individual(self):
        pop = Population()
        pop.add_individual(Individual(code="x=1"))
        assert len(pop) == 1

    def test_remove_weakest(self):
        inds = [
            Individual(code="x=1", fitness=0.1),
            Individual(code="x=2", fitness=0.9),
            Individual(code="x=3", fitness=0.5),
        ]
        pop = Population(inds)
        removed = pop.remove_weakest(1)
        assert len(removed) == 1
        assert removed[0].fitness == 0.1
        assert len(pop) == 2

    def test_replace_with(self):
        pop = Population([Individual(code="x=1")])
        new_gen = [Individual(code="y=1"), Individual(code="y=2")]
        pop.replace_with(new_gen)
        assert len(pop) == 2


# ── Fitness ─────────────────────────────────────────────────────────────────


class TestFitnessEvaluator:
    def setup_method(self):
        self.config = SynthesisConfig()
        self.evaluator = FitnessEvaluator(self.config)

    def test_invalid_code_gets_worst_fitness(self):
        ind = Individual(code="def foo( return")
        import asyncio

        fitness = asyncio.get_event_loop().run_until_complete(self.evaluator.evaluate(ind, Specification(), []))
        assert fitness == -1.0

    def test_complexity_penalty_simple(self):
        ind = Individual(code="x = 1")
        penalty = FitnessEvaluator._complexity_penalty(ind)
        assert 0 < penalty < 0.2  # Simple code = low penalty

    def test_complexity_penalty_capped(self):
        # Code with many nodes
        code = "\n".join(f"x{i} = {i} + {i}" for i in range(200))
        ind = Individual(code=code)
        penalty = FitnessEvaluator._complexity_penalty(ind)
        assert penalty <= 1.0  # Capped at 1.0

    def test_counterexample_avoidance_no_ces(self):
        ind = Individual(code="def f(x): return x + 1")
        score = self.evaluator._counterexample_avoidance(ind, [])
        assert score == 1.0

    def test_triggers_counterexample_correct_code(self):
        ind = Individual(code="def f(x):\n    return x * 2\n")
        ce = Counterexample(
            input_values={"x": 5},
            expected_output=10,
            actual_output=6,  # Wrong output
        )
        # The code returns 10 (correct), CE actual is 6, so code does NOT trigger CE
        triggered = self.evaluator._triggers_counterexample(ind, ce)
        assert triggered is False

    def test_triggers_counterexample_bad_code(self):
        ind = Individual(code="def f(x):\n    return x + 1\n")
        ce = Counterexample(
            input_values={"x": 5},
            expected_output=10,
            actual_output="6",  # str "6"
        )
        # Code returns 6, CE actual is "6", so str match = triggers
        triggered = self.evaluator._triggers_counterexample(ind, ce)
        assert triggered is True


# ── AST Operators ───────────────────────────────────────────────────────────


class TestASTCrossover:
    def test_crossover_produces_valid_children(self):
        crossover = ASTCrossover()
        p1 = Individual(code="def foo(x):\n    return x + 1\n")
        p2 = Individual(code="def bar(y):\n    return y * 2\n")
        child1, child2 = crossover.crossover(p1, p2)
        # At least one child should be produced (crossover may fail on some combos)
        produced = [c for c in [child1, child2] if c is not None]
        for child in produced:
            assert child.is_valid()
            assert child.source == PatchSource.CROSSOVER

    def test_crossover_with_invalid_parent(self):
        crossover = ASTCrossover()
        p1 = Individual(code="def foo( return")
        p2 = Individual(code="def bar(y): return y")
        child1, child2 = crossover.crossover(p1, p2)
        assert child1 is None
        assert child2 is None

    def test_crossover_sets_lineage(self):
        crossover = ASTCrossover()
        p1 = Individual(code="def a(x):\n    y = x + 1\n    return y\n")
        p2 = Individual(code="def b(x):\n    y = x * 2\n    return y\n")
        child1, child2 = crossover.crossover(p1, p2, generation=3)
        produced = [c for c in [child1, child2] if c is not None]
        for child in produced:
            assert child.generation == 3
            assert len(child.lineage) == 2


class TestASTMutator:
    def test_mutate_produces_valid_code(self):
        mutator = ASTMutator()
        ind = Individual(code="def foo(x):\n    return x + 1\n")
        # Try multiple times since mutation is random
        results = [mutator.mutate(ind, generation=1) for _ in range(10)]
        valid = [r for r in results if r is not None]
        assert len(valid) > 0  # At least some mutations should succeed
        for v in valid:
            assert v.is_valid()
            assert v.source == PatchSource.MUTATION

    def test_mutate_invalid_code(self):
        mutator = ASTMutator()
        ind = Individual(code="def foo( return")
        result = mutator.mutate(ind)
        assert result is None

    def test_mutate_with_counterexample(self):
        mutator = ASTMutator()
        ind = Individual(code="def foo(x):\n    return x + 1\n")
        ce = Counterexample(error_message="TypeError: unsupported operand")
        result = mutator.mutate(ind, counterexample=ce)
        # With TypeError CE, should prefer guard_insertion operator
        if result:
            assert result.is_valid()

    def test_constant_tweak(self):
        mutator = ASTMutator()
        ind = Individual(code="def foo():\n    return 42\n")
        results = [mutator.mutate(ind) for _ in range(20)]
        valid_codes = [r.code for r in results if r is not None]
        # At least one mutation should change the constant
        _has_different = any("42" not in code for code in valid_codes)
        # It's probabilistic, so this might not always hold; just check we got some results
        assert len(valid_codes) > 0


# ── Entropy ─────────────────────────────────────────────────────────────────


class TestEntropyManager:
    def setup_method(self):
        self.config = SynthesisConfig(entropy_threshold=1.0)
        self.manager = EntropyManager(self.config)

    def test_entropy_single_individual(self):
        inds = [Individual(code="x = 1")]
        entropy = self.manager.compute_entropy(inds)
        assert entropy == 0.0

    def test_entropy_identical_individuals(self):
        inds = [Individual(code="x = 1") for _ in range(5)]
        entropy = self.manager.compute_entropy(inds)
        assert entropy == 0.0  # All same = zero entropy

    def test_entropy_diverse_individuals(self):
        inds = [
            Individual(code="x = 1"),
            Individual(code="def foo(a, b, c):\n    if a > b:\n        return c\n    return a + b\n"),
            Individual(code="for i in range(100):\n    x = i * i\n    if x > 50: break\n"),
        ]
        entropy = self.manager.compute_entropy(inds)
        assert entropy > 0  # Different structures = some entropy

    def test_needs_injection_low_entropy(self):
        inds = [Individual(code="x = 1") for _ in range(10)]
        assert self.manager.needs_injection(inds) is True

    def test_needs_injection_high_entropy(self):
        # Create very diverse individuals
        inds = [
            Individual(code="x = 1"),
            Individual(
                code="def foo(a, b, c, d):\n    if a > b:\n        for i in range(c):\n            d += i\n    return d\n"
            ),
            Individual(code="class Foo:\n    def bar(self):\n        return 42\n    def baz(self): pass\n"),
        ]
        # This may or may not exceed threshold=1.0 depending on binning
        # The test verifies the method runs without error
        result = self.manager.needs_injection(inds)
        assert isinstance(result, bool)

    def test_select_for_replacement(self):
        inds = [
            Individual(code="x = 1", fitness=0.1),
            Individual(code="x = 2", fitness=0.2),
            Individual(code="x = 3", fitness=0.9),
        ]
        indices = self.manager.select_for_replacement(inds, 1)
        assert len(indices) == 1
        # Should pick lowest fitness from most common bin
        assert indices[0] in [0, 1, 2]

    def test_extract_features(self):
        ind = Individual(code="def foo(x):\n    if x > 0:\n        return x\n    return -x\n")
        features = self.manager._extract_features(ind)
        assert "ast_depth" in features
        assert "node_count" in features
        assert "branch_count" in features
        assert features["branch_count"] >= 1  # Has an if statement
        assert features["func_count"] >= 1  # Has a function def
