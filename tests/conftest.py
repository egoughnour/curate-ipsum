"""
Shared test fixtures and pytest configuration.

Markers:
    @pytest.mark.integration  — Requires real external services (Docker, model downloads)
    @pytest.mark.docker       — Requires Docker daemon running
    @pytest.mark.embedding    — Requires sentence-transformers model downloadable

Run stringent tests:
    pytest -m integration             # all integration tests
    pytest -m docker                  # only Docker tests
    pytest -m embedding               # only embedding model tests
    pytest -m "not integration"       # skip all integration tests (fast CI)
"""

import os
import shutil
import subprocess
from typing import Optional

import pytest


def _docker_available() -> bool:
    """Check if Docker CLI is on PATH and the daemon responds."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _embedding_model_available() -> bool:
    """Check if all-MiniLM-L6-v2 can be loaded (already cached or downloadable)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = model.encode(["test"])
        return vec.shape[1] == 384
    except Exception:
        return False


# Cache the checks at module level so they run once per session
_DOCKER_OK: Optional[bool] = None
_EMBEDDING_OK: Optional[bool] = None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: requires external services (Docker, model downloads)")
    config.addinivalue_line("markers", "docker: requires Docker daemon running")
    config.addinivalue_line("markers", "embedding: requires sentence-transformers model available")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests whose infrastructure requirements are not met."""
    global _DOCKER_OK, _EMBEDDING_OK

    # Only evaluate once
    if _DOCKER_OK is None:
        _DOCKER_OK = _docker_available()
    if _EMBEDDING_OK is None:
        _EMBEDDING_OK = _embedding_model_available()

    skip_docker = pytest.mark.skip(reason="Docker daemon not available")
    skip_embedding = pytest.mark.skip(reason="Embedding model not available (all-MiniLM-L6-v2)")

    for item in items:
        if "docker" in item.keywords and not _DOCKER_OK:
            item.add_marker(skip_docker)
        if "embedding" in item.keywords and not _EMBEDDING_OK:
            item.add_marker(skip_embedding)
        # integration implies both
        if "integration" in item.keywords:
            if not _DOCKER_OK:
                item.add_marker(skip_docker)
            if not _EMBEDDING_OK:
                item.add_marker(skip_embedding)
