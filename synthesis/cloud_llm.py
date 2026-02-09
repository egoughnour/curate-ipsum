"""
Cloud LLM client: Anthropic (Claude) and OpenAI (GPT) backends.

Uses httpx for async HTTP. API key from environment variable
CURATE_IPSUM_LLM_API_KEY or passed directly.

Decision: D-012 — abstract LLM client with cloud/local/mock backends.
"""

from __future__ import annotations

import logging
import os
import re
import time

from synthesis.llm_client import LLMClient

LOG = logging.getLogger("synthesis.cloud_llm")

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


class CloudLLMClient(LLMClient):
    """
    Cloud LLM backend using Anthropic or OpenAI APIs.

    Supports:
    - anthropic: Claude models via messages API
    - openai: GPT models via chat completions API
    """

    def __init__(
        self,
        api_key: str | None = None,
        provider: str = "anthropic",  # "anthropic" or "openai"
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str | None = None,
        max_retries: int = 3,
        requests_per_second: float = 5.0,
    ) -> None:
        if httpx is None:
            raise ImportError("httpx is required for cloud LLM. Install with: pip install 'curate-ipsum[synthesis]'")

        self._api_key = api_key or os.environ.get("CURATE_IPSUM_LLM_API_KEY", "")
        if not self._api_key:
            raise ValueError("API key required. Set CURATE_IPSUM_LLM_API_KEY or pass api_key=.")

        self._provider = provider
        self._model = model
        self._max_retries = max_retries
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._total_cost_estimate = 0.0

        if base_url:
            self._base_url = base_url
        elif provider == "anthropic":
            self._base_url = "https://api.anthropic.com/v1"
        else:
            self._base_url = "https://api.openai.com/v1"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=60.0,
        )

    async def generate_candidates(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.8,
    ) -> list[str]:
        """Generate n candidates by making n API calls (one per candidate)."""
        candidates: list[str] = []

        for _ in range(n):
            # Rate limiting
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                import asyncio

                await asyncio.sleep(self._min_interval - elapsed)

            response_text = await self._call_api(prompt, temperature)
            self._last_request_time = time.monotonic()

            if response_text:
                code = self._extract_code(response_text)
                if code:
                    candidates.append(code)

        LOG.info(
            "Cloud LLM generated %d/%d candidates (est. cost: $%.4f)",
            len(candidates),
            n,
            self._total_cost_estimate,
        )
        return candidates

    async def _call_api(self, prompt: str, temperature: float) -> str:
        """Make a single API call with retry."""
        for attempt in range(self._max_retries):
            try:
                if self._provider == "anthropic":
                    return await self._call_anthropic(prompt, temperature)
                else:
                    return await self._call_openai(prompt, temperature)
            except Exception as exc:
                wait = 2**attempt
                LOG.warning(
                    "API call failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1,
                    self._max_retries,
                    exc,
                    wait,
                )
                import asyncio

                await asyncio.sleep(wait)
        return ""

    async def _call_anthropic(self, prompt: str, temperature: float) -> str:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self._model,
            "max_tokens": 2000,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = await self._client.post("/messages", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        # Estimate cost (rough: $0.003/1K input + $0.015/1K output for Sonnet)
        self._total_cost_estimate += 0.02
        return data.get("content", [{}])[0].get("text", "")

    async def _call_openai(self, prompt: str, temperature: float) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "temperature": temperature,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = await self._client.post("/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        self._total_cost_estimate += 0.02
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from LLM response, stripping markdown fences."""
        # Try to find ```python ... ``` blocks
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        # If no fences, return the whole text (likely raw code)
        return text.strip()

    @property
    def total_cost_estimate(self) -> float:
        return self._total_cost_estimate

    async def close(self) -> None:
        await self._client.aclose()
