"""Tests for LLM client abstraction and prompt construction."""

import pytest
from synthesis.llm_client import MockLLMClient, build_synthesis_prompt
from synthesis.models import Counterexample, Specification


class TestMockLLMClient:
    @pytest.mark.asyncio
    async def test_default_candidates(self):
        client = MockLLMClient()
        candidates = await client.generate_candidates("test prompt", n=3)
        assert len(candidates) == 3
        for c in candidates:
            assert "def patched_func" in c

    @pytest.mark.asyncio
    async def test_canned_responses(self):
        client = MockLLMClient(responses=["def foo(): return 1", "def bar(): return 2"])
        candidates = await client.generate_candidates("test", n=4)
        assert len(candidates) == 4
        assert candidates[0] == "def foo(): return 1"
        assert candidates[1] == "def bar(): return 2"
        assert candidates[2] == "def foo(): return 1"  # Cycles

    @pytest.mark.asyncio
    async def test_call_count(self):
        client = MockLLMClient()
        assert client.call_count == 0
        await client.generate_candidates("test", n=1)
        assert client.call_count == 1
        await client.generate_candidates("test", n=1)
        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        client = MockLLMClient()
        await client.close()  # Should not raise


class TestBuildSynthesisPrompt:
    def test_minimal_prompt(self):
        spec = Specification()
        prompt = build_synthesis_prompt(spec)
        assert "Generate a Python function" in prompt

    def test_includes_original_code(self):
        spec = Specification(original_code="def add(a, b): return a + b")
        prompt = build_synthesis_prompt(spec)
        assert "def add(a, b): return a + b" in prompt
        assert "Original Code" in prompt

    def test_includes_preconditions(self):
        spec = Specification(preconditions=["x > 0", "y is not None"])
        prompt = build_synthesis_prompt(spec)
        assert "Preconditions" in prompt
        assert "x > 0" in prompt
        assert "y is not None" in prompt

    def test_includes_postconditions(self):
        spec = Specification(postconditions=["result >= 0"])
        prompt = build_synthesis_prompt(spec)
        assert "Postconditions" in prompt
        assert "result >= 0" in prompt

    def test_includes_mutant_ids(self):
        spec = Specification(surviving_mutant_ids=["m1", "m2"])
        prompt = build_synthesis_prompt(spec)
        assert "m1" in prompt
        assert "m2" in prompt
        assert "Kill surviving mutants" in prompt

    def test_includes_test_commands(self):
        spec = Specification(test_commands=["pytest tests/test_add.py"])
        prompt = build_synthesis_prompt(spec)
        assert "pytest tests/test_add.py" in prompt

    def test_includes_counterexamples(self):
        ces = [
            Counterexample(input_values={"x": -1}, expected_output=0, actual_output=-1, error_message="negative input"),
        ]
        spec = Specification()
        prompt = build_synthesis_prompt(spec, counterexamples=ces)
        assert "counterexamples" in prompt.lower()
        assert "negative input" in prompt

    def test_limits_counterexamples_to_5(self):
        ces = [Counterexample(error_message=f"err_{i}") for i in range(10)]
        spec = Specification()
        prompt = build_synthesis_prompt(spec, counterexamples=ces)
        # Should include last 5, not all 10
        assert "err_9" in prompt
        assert "err_5" in prompt
        # First ones may not appear
        assert "err_0" not in prompt

    def test_includes_context_code(self):
        spec = Specification()
        prompt = build_synthesis_prompt(spec, context_code="import math\n")
        assert "import math" in prompt
        assert "Context" in prompt


class TestCloudLLMExtractCode:
    def test_extract_from_fenced_block(self):
        from synthesis.cloud_llm import CloudLLMClient
        text = "Here's the code:\n```python\ndef foo():\n    return 42\n```\n"
        result = CloudLLMClient._extract_code(text)
        assert "def foo():" in result
        assert "return 42" in result

    def test_extract_from_plain_text(self):
        from synthesis.cloud_llm import CloudLLMClient
        text = "def foo():\n    return 42\n"
        result = CloudLLMClient._extract_code(text)
        assert "def foo():" in result

    def test_extract_from_generic_fence(self):
        from synthesis.cloud_llm import CloudLLMClient
        text = "```\ndef bar(): pass\n```"
        result = CloudLLMClient._extract_code(text)
        assert "def bar(): pass" in result


class TestCloudLLMValidation:
    def test_requires_api_key(self):
        """CloudLLMClient should raise ValueError without API key."""
        import os
        old = os.environ.pop("CURATE_IPSUM_LLM_API_KEY", None)
        try:
            pytest.importorskip("httpx")
            from synthesis.cloud_llm import CloudLLMClient
            with pytest.raises(ValueError, match="API key"):
                CloudLLMClient(api_key="")
        finally:
            if old:
                os.environ["CURATE_IPSUM_LLM_API_KEY"] = old


class TestLocalLLMExtractCode:
    def test_extract_from_fenced_block(self):
        from synthesis.local_llm import LocalLLMClient
        text = "```python\ndef baz(): return 1\n```"
        result = LocalLLMClient._extract_code(text)
        assert "def baz(): return 1" in result

    def test_extract_from_plain_text(self):
        from synthesis.local_llm import LocalLLMClient
        text = "def baz(): return 1"
        result = LocalLLMClient._extract_code(text)
        assert "def baz(): return 1" in result
