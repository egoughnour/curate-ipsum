"""
Local LLM client: Ollama backend.

Connects to a locally running Ollama instance at http://localhost:11434.
Default model: codellama:7b.

Decision: D-012 — abstract LLM client with cloud/local/mock backends.
"""

from __future__ import annotations

import logging
import re

from synthesis.llm_client import LLMClient

LOG = logging.getLogger("synthesis.local_llm")

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


class LocalLLMClient(LLMClient):
    """
    Local LLM backend using Ollama's HTTP API.

    Requires Ollama to be running locally: https://ollama.ai
    Default model: codellama:7b (good for code generation, runs on 8GB+ GPU).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "codellama:7b",
        timeout: float = 120.0,
    ) -> None:
        if httpx is None:
            raise ImportError("httpx is required for local LLM. Install with: pip install 'curate-ipsum[synthesis]'")

        self._base_url = base_url
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
        )
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        if self._available is not None:
            return self._available
        try:
            resp = await self._client.get("/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our model (or a prefix match) is available
                self._available = any(self._model in name for name in model_names)
                if not self._available:
                    LOG.warning(
                        "Ollama running but model '%s' not found. Available: %s. Pull with: ollama pull %s",
                        self._model,
                        model_names,
                        self._model,
                    )
                return self._available
        except Exception as exc:
            LOG.warning("Ollama not reachable at %s: %s", self._base_url, exc)
        self._available = False
        return False

    async def generate_candidates(
        self,
        prompt: str,
        n: int = 5,
        temperature: float = 0.8,
    ) -> list[str]:
        if not await self.is_available():
            LOG.error(
                "Ollama not available. Start with 'ollama serve' and ensure '%s' is pulled.",
                self._model,
            )
            return []

        candidates: list[str] = []

        for i in range(n):
            try:
                resp = await self._client.post(
                    "/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": 2000,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("response", "")
                code = self._extract_code(text)
                if code:
                    candidates.append(code)
            except Exception as exc:
                LOG.warning("Ollama generation %d/%d failed: %s", i + 1, n, exc)

        LOG.info("Local LLM generated %d/%d candidates", len(candidates), n)
        return candidates

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from Ollama response."""
        # Try markdown fences first
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return text.strip()

    async def close(self) -> None:
        await self._client.aclose()
