"""Configuration management for Quick Verification.

Loads settings from environment variables with sensible defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class VerificationConfig:
    """Verification subsystem configuration."""
    backend: str = "mock"  # "mock", "z3", "angr-docker"
    angr_image: str = "angr/angr"
    artifacts_dir: str = "artifacts/verify"
    runner_script: str = ""  # empty = auto-discover

    @classmethod
    def from_env(cls) -> "VerificationConfig":
        return cls(
            backend=os.getenv("VERIFY_BACKEND", "mock"),
            angr_image=os.getenv("ANGR_IMAGE", "angr/angr"),
            artifacts_dir=os.getenv("ANGR_ARTIFACTS_DIR", "artifacts/verify"),
            runner_script=os.getenv("ANGR_RUNNER_SCRIPT", ""),
        )


@dataclass
class RAGConfig:
    """RAG subsystem configuration."""
    vector_store_backend: str = "chroma"  # "chroma"
    chroma_persist_dir: str = "./data/chroma"
    chroma_host: str = ""  # empty = embedded mode
    chroma_port: int = 8000
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "code_nodes"
    top_k: int = 20
    expansion_hops: int = 1

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            vector_store_backend=os.getenv("VECTOR_STORE_BACKEND", "chroma"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
            chroma_host=os.getenv("CHROMA_HOST", ""),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            collection_name=os.getenv("CHROMA_COLLECTION", "code_nodes"),
            top_k=int(os.getenv("RAG_TOP_K", "20")),
            expansion_hops=int(os.getenv("RAG_EXPANSION_HOPS", "1")),
        )


@dataclass
class AppConfig:
    """Top-level application configuration."""
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            verification=VerificationConfig.from_env(),
            rag=RAGConfig.from_env(),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
