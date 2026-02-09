"""Tests for synthesis data models."""

import pytest

from synthesis.models import (
    CodePatch,
    Counterexample,
    Individual,
    LLMBackend,
    PatchSource,
    Specification,
    SynthesisConfig,
    SynthesisResult,
    SynthesisStatus,
)


class TestEnums:
    def test_synthesis_status_values(self):
        assert SynthesisStatus.SUCCESS.value == "success"
        assert SynthesisStatus.FAILED.value == "failed"
        assert SynthesisStatus.TIMEOUT.value == "timeout"
        assert SynthesisStatus.CANCELLED.value == "cancelled"

    def test_patch_source_values(self):
        assert PatchSource.LLM.value == "llm"
        assert PatchSource.CROSSOVER.value == "crossover"
        assert PatchSource.MUTATION.value == "mutation"
        assert PatchSource.SEED.value == "seed"

    def test_llm_backend_values(self):
        assert LLMBackend.CLOUD.value == "cloud"
        assert LLMBackend.LOCAL.value == "local"
        assert LLMBackend.MOCK.value == "mock"


class TestSynthesisConfig:
    def test_default_config(self):
        config = SynthesisConfig()
        assert config.population_size == 20
        assert config.max_iterations == 100
        assert config.mutation_rate == 0.3
        assert config.crossover_rate == 0.7
        assert config.elite_ratio == 0.1
        assert config.llm_backend == "mock"

    def test_custom_config(self):
        config = SynthesisConfig(population_size=50, mutation_rate=0.5)
        assert config.population_size == 50
        assert config.mutation_rate == 0.5

    def test_invalid_mutation_rate(self):
        with pytest.raises(ValueError, match="mutation_rate"):
            SynthesisConfig(mutation_rate=1.5)

    def test_invalid_crossover_rate(self):
        with pytest.raises(ValueError, match="crossover_rate"):
            SynthesisConfig(crossover_rate=-0.1)

    def test_invalid_elite_ratio(self):
        with pytest.raises(ValueError, match="elite_ratio"):
            SynthesisConfig(elite_ratio=2.0)

    def test_invalid_population_size(self):
        with pytest.raises(ValueError, match="population_size"):
            SynthesisConfig(population_size=1)

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations"):
            SynthesisConfig(max_iterations=0)

    def test_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            SynthesisConfig(top_k=0)

    def test_boundary_values_valid(self):
        # All boundaries should be valid
        config = SynthesisConfig(
            mutation_rate=0.0, crossover_rate=1.0, elite_ratio=0.0, population_size=2, max_iterations=1, top_k=1
        )
        assert config.mutation_rate == 0.0
        assert config.crossover_rate == 1.0


class TestIndividual:
    def test_creation(self):
        ind = Individual(code="x = 1")
        assert ind.code == "x = 1"
        assert ind.fitness == 0.0
        assert ind.generation == 0
        assert ind.source == PatchSource.SEED
        assert ind.id  # UUID assigned

    def test_is_valid_good_code(self):
        ind = Individual(code="def foo(x):\n    return x + 1\n")
        assert ind.is_valid() is True

    def test_is_valid_bad_code(self):
        ind = Individual(code="def foo(x) return x")
        assert ind.is_valid() is False

    def test_is_valid_empty_code(self):
        ind = Individual(code="")
        assert ind.is_valid() is True  # Empty string is valid Python

    def test_unique_ids(self):
        a = Individual()
        b = Individual()
        assert a.id != b.id


class TestCodePatch:
    def test_to_dict(self):
        patch = CodePatch(code="return 42", source=PatchSource.LLM, region_id="r1")
        d = patch.to_dict()
        assert d["code"] == "return 42"
        assert d["source"] == "llm"
        assert d["region_id"] == "r1"

    def test_defaults(self):
        patch = CodePatch(code="pass")
        assert patch.source == PatchSource.LLM
        assert patch.diff == ""
        assert patch.region_id == ""


class TestCounterexample:
    def test_to_dict(self):
        ce = Counterexample(
            input_values={"x": 5},
            expected_output=10,
            actual_output=6,
            error_message="wrong result",
        )
        d = ce.to_dict()
        assert d["input_values"] == {"x": 5}
        assert d["expected_output"] == "10"
        assert d["actual_output"] == "6"
        assert d["error_message"] == "wrong result"
        assert d["id"]  # UUID assigned


class TestSpecification:
    def test_defaults(self):
        spec = Specification()
        assert spec.target_region == ""
        assert spec.test_commands == []
        assert spec.surviving_mutant_ids == []

    def test_with_values(self):
        spec = Specification(
            target_region="func_add",
            test_commands=["pytest tests/"],
            surviving_mutant_ids=["m1", "m2"],
            preconditions=["x > 0"],
        )
        assert spec.target_region == "func_add"
        assert len(spec.test_commands) == 1
        assert len(spec.surviving_mutant_ids) == 2


class TestSynthesisResult:
    def test_to_dict_no_patch(self):
        result = SynthesisResult(status=SynthesisStatus.FAILED, iterations=10)
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["patch"] is None
        assert d["iterations"] == 10

    def test_to_dict_with_patch(self):
        patch = CodePatch(code="return 42")
        result = SynthesisResult(
            status=SynthesisStatus.SUCCESS,
            patch=patch,
            iterations=5,
            counterexamples_resolved=3,
        )
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["patch"]["code"] == "return 42"
        assert d["counterexamples_resolved"] == 3

    def test_defaults(self):
        result = SynthesisResult()
        assert result.status == SynthesisStatus.FAILED
        assert result.iterations == 0
        assert result.fitness_history == []
