"""Tests for the CEGIS synthesis engine."""

import asyncio
import pytest
from synthesis.cegis import CEGISEngine
from synthesis.llm_client import MockLLMClient
from synthesis.models import (
    Specification,
    SynthesisConfig,
    SynthesisResult,
    SynthesisStatus,
)


class TestCEGISBasic:
    @pytest.mark.asyncio
    async def test_synthesis_with_mock_llm(self):
        """Basic synthesis run with MockLLMClient should complete."""
        config = SynthesisConfig(
            max_iterations=10,
            population_size=5,
            top_k=5,
        )
        client = MockLLMClient(responses=[
            "def patched_func(x):\n    return x + 1\n",
            "def patched_func(x):\n    return x * 2\n",
        ])
        engine = CEGISEngine(config, client)
        spec = Specification()

        result = await engine.synthesize(spec)

        assert isinstance(result, SynthesisResult)
        assert result.status in (SynthesisStatus.SUCCESS, SynthesisStatus.FAILED)
        assert result.iterations > 0
        assert result.duration_ms > 0
        assert len(result.fitness_history) > 0

    @pytest.mark.asyncio
    async def test_all_invalid_candidates_fails(self):
        """If LLM returns only syntactically invalid candidates, synthesis should fail."""
        config = SynthesisConfig(max_iterations=5, population_size=3, top_k=3)
        client = MockLLMClient(responses=[
            "def foo( return",  # Invalid syntax
            "class { broken",  # Invalid syntax
        ])
        engine = CEGISEngine(config, client)
        spec = Specification()

        result = await engine.synthesize(spec)

        assert result.status == SynthesisStatus.FAILED
        assert "no syntactically valid" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_max_iterations_exhausted(self):
        """Synthesis should fail after max_iterations with appropriate message."""
        config = SynthesisConfig(
            max_iterations=3,
            population_size=4,
            top_k=4,
        )
        # Provide candidates that won't satisfy spec easily
        client = MockLLMClient(responses=[
            "x = 1\n",
            "y = 2\n",
        ])
        engine = CEGISEngine(config, client)
        spec = Specification(
            test_commands=["pytest nonexistent/"],  # Will fail
            working_directory="/tmp",  # Valid but tests won't exist
        )

        result = await engine.synthesize(spec)

        # Should be FAILED or TIMEOUT (depending on test execution)
        assert result.status in (SynthesisStatus.FAILED, SynthesisStatus.TIMEOUT)
        assert result.iterations <= 3

    @pytest.mark.asyncio
    async def test_cancellation(self):
        """Cancelling mid-run should return CANCELLED status."""
        from synthesis.llm_client import LLMClient

        class SlowMockClient(LLMClient):
            """LLM client that sleeps, giving time for cancellation."""
            async def generate_candidates(self, prompt, n=5, temperature=0.8):
                await asyncio.sleep(0.2)  # Slow enough for cancel to fire
                return [f"def f(x): return x + {i}\n" for i in range(n)]

        config = SynthesisConfig(
            max_iterations=100,
            population_size=5,
            top_k=5,
        )
        client = SlowMockClient()
        engine = CEGISEngine(config, client)
        spec = Specification()

        # Cancel after a short delay — during the slow LLM call
        async def cancel_soon():
            await asyncio.sleep(0.05)
            engine.cancel()

        cancel_task = asyncio.create_task(cancel_soon())
        result = await engine.synthesize(spec)
        await cancel_task

        # Engine should have been cancelled after the LLM call completed
        # but before completing all 100 iterations
        assert result.status == SynthesisStatus.CANCELLED
        assert "cancelled" in result.error_message.lower()
        assert result.iterations < 100


class TestCEGISPopulationEvolution:
    @pytest.mark.asyncio
    async def test_fitness_history_tracked(self):
        """Fitness history should have one entry per iteration."""
        config = SynthesisConfig(
            max_iterations=5,
            population_size=5,
            top_k=5,
        )
        client = MockLLMClient()
        engine = CEGISEngine(config, client)
        spec = Specification()

        result = await engine.synthesize(spec)

        assert len(result.fitness_history) == result.iterations
        # All fitness values should be numeric
        for f in result.fitness_history:
            assert isinstance(f, float)

    @pytest.mark.asyncio
    async def test_total_candidates_tracked(self):
        """Total candidates evaluated should be > 0."""
        config = SynthesisConfig(max_iterations=3, population_size=5, top_k=5)
        client = MockLLMClient()
        engine = CEGISEngine(config, client)
        spec = Specification()

        result = await engine.synthesize(spec)

        assert result.total_candidates_evaluated > 0


class TestCEGISTimeout:
    @pytest.mark.asyncio
    async def test_synthesis_timeout(self):
        """Synthesis should respect timeout."""
        config = SynthesisConfig(
            max_iterations=10000,  # Very high
            population_size=5,
            top_k=5,
            synthesis_timeout_seconds=0.1,  # Very short timeout
        )
        client = MockLLMClient()
        engine = CEGISEngine(config, client)
        spec = Specification()

        result = await engine.synthesize(spec)

        assert result.status == SynthesisStatus.TIMEOUT
        assert result.iterations < 10000  # Didn't run all iterations


class TestCEGISResultFormat:
    @pytest.mark.asyncio
    async def test_result_serializable(self):
        """SynthesisResult.to_dict() should produce a clean dict."""
        config = SynthesisConfig(max_iterations=3, population_size=3, top_k=3)
        client = MockLLMClient()
        engine = CEGISEngine(config, client)
        spec = Specification()

        result = await engine.synthesize(spec)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "status" in d
        assert "iterations" in d
        assert "fitness_history" in d
        assert "duration_ms" in d
        assert isinstance(d["fitness_history"], list)
