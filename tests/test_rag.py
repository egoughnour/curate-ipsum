"""Tests for RAG subsystem — vector store, embedding provider, search pipeline."""

from __future__ import annotations

from typing import Any

import pytest

# ── Mock GraphStore for testing graph expansion ──────────────────────────────


class MockGraphStoreForRAG:
    """Minimal mock implementing the GraphStore interface methods RAG uses."""

    def __init__(self, neighbors=None, nodes=None):
        self._neighbors = neighbors or {}  # {(node_id, direction): [ids]}
        self._nodes = nodes or {}  # {node_id: {data}}

    def get_neighbors(
        self, node_id: str, project_id: str, direction: str = "outgoing", edge_kind: str | None = None
    ) -> list[str]:
        return self._neighbors.get((node_id, direction), [])

    def get_node(self, node_id: str, project_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestVectorStore:
    def test_chroma_add_and_count(self):
        pytest.importorskip("chromadb")
        from curate_ipsum.rag.vector_store import ChromaVectorStore, VectorDocument

        store = ChromaVectorStore(collection_name="test_add_count")
        store.add(
            [
                VectorDocument(id="fn1", text="def hello(): pass", embedding=[1.0, 0.0, 0.0], metadata={"kind": "fn"}),
                VectorDocument(id="fn2", text="def world(): pass", embedding=[0.0, 1.0, 0.0], metadata={"kind": "fn"}),
            ]
        )
        assert store.count() == 2

    def test_chroma_search(self):
        pytest.importorskip("chromadb")
        from curate_ipsum.rag.vector_store import ChromaVectorStore, VectorDocument

        store = ChromaVectorStore(collection_name="test_search")
        store.add(
            [
                VectorDocument(id="fn1", text="def hello(): pass", embedding=[1.0, 0.0, 0.0], metadata={"kind": "fn"}),
                VectorDocument(
                    id="fn2", text="def world(): return 42", embedding=[0.0, 1.0, 0.0], metadata={"kind": "fn"}
                ),
            ]
        )
        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].id == "fn1"
        assert results[0].score > results[1].score

    def test_chroma_delete(self):
        pytest.importorskip("chromadb")
        from curate_ipsum.rag.vector_store import ChromaVectorStore, VectorDocument

        store = ChromaVectorStore(collection_name="test_delete")
        store.add(
            [
                VectorDocument(id="fn1", text="hello", embedding=[1.0, 0.0, 0.0], metadata={"kind": "fn"}),
                VectorDocument(id="fn2", text="world", embedding=[0.0, 1.0, 0.0], metadata={"kind": "fn"}),
            ]
        )
        assert store.count() == 2
        store.delete(["fn1"])
        assert store.count() == 1

    def test_chroma_empty_search(self):
        pytest.importorskip("chromadb")
        from curate_ipsum.rag.vector_store import ChromaVectorStore

        store = ChromaVectorStore(collection_name="test_empty_search")
        results = store.search([1.0, 0.0], top_k=5)
        assert results == []

    def test_build_factory(self):
        pytest.importorskip("chromadb")
        from curate_ipsum.rag.vector_store import build_vector_store

        store = build_vector_store("chroma", collection_name="test_factory")
        assert store.count() == 0

    def test_factory_unknown_raises(self):
        from curate_ipsum.rag.vector_store import build_vector_store

        with pytest.raises(ValueError, match="Unknown vector store backend"):
            build_vector_store("nonexistent")


class TestEmbeddingProvider:
    def test_mock_embedding(self):
        from curate_ipsum.rag.embedding_provider import MockEmbeddingProvider

        provider = MockEmbeddingProvider(dim=128)
        vecs = provider.embed(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 128
        assert all(v == 0.0 for v in vecs[0])
        assert provider.dimension() == 128

    def test_mock_empty(self):
        from curate_ipsum.rag.embedding_provider import MockEmbeddingProvider

        provider = MockEmbeddingProvider()
        vecs = provider.embed([])
        assert vecs == []


class TestRAGPipeline:
    def test_search_returns_results(self):
        pytest.importorskip("chromadb")
        from curate_ipsum.rag.embedding_provider import MockEmbeddingProvider
        from curate_ipsum.rag.search import RAGConfig, RAGPipeline
        from curate_ipsum.rag.vector_store import ChromaVectorStore, VectorDocument

        store = ChromaVectorStore(collection_name="test_pipeline_search")
        store.add(
            [
                VectorDocument(id="fn1", text="validate input", embedding=[0.0] * 384, metadata={"kind": "fn"}),
                VectorDocument(id="fn2", text="process data", embedding=[0.0] * 384, metadata={"kind": "fn"}),
            ]
        )

        embedder = MockEmbeddingProvider(dim=384)
        pipeline = RAGPipeline(store, embedder, config=RAGConfig(vector_top_k=5))
        results = pipeline.search("validate")

        assert len(results) >= 1
        assert all(hasattr(r, "node_id") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_pack_context(self):
        pytest.importorskip("chromadb")
        from curate_ipsum.rag.embedding_provider import MockEmbeddingProvider
        from curate_ipsum.rag.search import RAGConfig, RAGPipeline, RAGResult
        from curate_ipsum.rag.vector_store import ChromaVectorStore

        store = ChromaVectorStore(collection_name="test_pack_context")
        embedder = MockEmbeddingProvider()
        pipeline = RAGPipeline(store, embedder, config=RAGConfig(max_context_tokens=100))

        results = [
            RAGResult(node_id="fn1", text="def foo(): pass", score=0.9, source="vector"),
            RAGResult(node_id="fn2", text="def bar(): return 1", score=0.7, source="vector"),
        ]
        packed = pipeline.pack_context(results)
        assert "fn1" in packed
        assert "fn2" in packed

    def test_graph_expansion(self):
        """Pipeline should expand results using GraphStore neighbors."""
        pytest.importorskip("chromadb")
        from curate_ipsum.rag.embedding_provider import MockEmbeddingProvider
        from curate_ipsum.rag.search import RAGConfig, RAGPipeline
        from curate_ipsum.rag.vector_store import ChromaVectorStore, VectorDocument

        store = ChromaVectorStore(collection_name="test_graph_expansion")
        store.add(
            [
                VectorDocument(id="fn1", text="validate input", embedding=[0.0] * 384, metadata={"kind": "fn"}),
            ]
        )

        mock_gs = MockGraphStoreForRAG(
            neighbors={
                ("fn1", "outgoing"): ["fn2", "fn3"],
                ("fn1", "incoming"): ["fn0"],
            },
            nodes={
                "fn2": {"id": "fn2", "label": "process_data"},
                "fn3": {"id": "fn3", "label": "send_response"},
                "fn0": {"id": "fn0", "label": "main_handler"},
            },
        )

        embedder = MockEmbeddingProvider(dim=384)
        pipeline = RAGPipeline(
            store,
            embedder,
            graph_store=mock_gs,
            config=RAGConfig(vector_top_k=5, expansion_hops=1),
        )
        results = pipeline.search("validate")

        node_ids = {r.node_id for r in results}
        # Should have fn1 from vector search + fn2, fn3, fn0 from graph expansion
        assert "fn1" in node_ids
        # At least some graph-expanded nodes
        graph_sources = [r for r in results if r.source.startswith("graph_")]
        assert len(graph_sources) >= 1


class TestCEGISVerificationIntegration:
    """Test that CEGIS engine accepts verification_backend parameter."""

    @pytest.mark.asyncio
    async def test_cegis_with_mock_verification(self):
        from curate_ipsum.synthesis.cegis import CEGISEngine
        from curate_ipsum.synthesis.llm_client import MockLLMClient
        from curate_ipsum.synthesis.models import Specification, SynthesisConfig, SynthesisStatus
        from curate_ipsum.verification.backends.mock import MockBackend

        config = SynthesisConfig(max_iterations=5, population_size=3, top_k=3)
        llm = MockLLMClient()
        vbackend = MockBackend(mode="no_ce")

        engine = CEGISEngine(config, llm, verification_backend=vbackend)
        spec = Specification()
        result = await engine.synthesize(spec)

        assert result.status in (SynthesisStatus.SUCCESS, SynthesisStatus.FAILED)

    @pytest.mark.asyncio
    async def test_cegis_with_rag_pipeline(self):
        """CEGIS should enrich prompts via RAG when pipeline is provided."""
        pytest.importorskip("chromadb")
        from curate_ipsum.rag.embedding_provider import MockEmbeddingProvider
        from curate_ipsum.rag.search import RAGConfig, RAGPipeline
        from curate_ipsum.rag.vector_store import ChromaVectorStore, VectorDocument
        from curate_ipsum.synthesis.cegis import CEGISEngine
        from curate_ipsum.synthesis.llm_client import MockLLMClient
        from curate_ipsum.synthesis.models import Specification, SynthesisConfig, SynthesisStatus

        store = ChromaVectorStore(collection_name="test_cegis_rag")
        store.add(
            [
                VectorDocument(
                    id="ctx1", text="def helper(): return 42", embedding=[0.0] * 384, metadata={"kind": "fn"}
                ),
            ]
        )
        embedder = MockEmbeddingProvider(dim=384)
        pipeline = RAGPipeline(store, embedder, config=RAGConfig(vector_top_k=3))

        config = SynthesisConfig(max_iterations=3, population_size=3, top_k=3)
        llm = MockLLMClient()
        engine = CEGISEngine(config, llm, rag_pipeline=pipeline)
        spec = Specification(original_code="def foo(): pass")
        result = await engine.synthesize(spec)

        assert result.status in (SynthesisStatus.SUCCESS, SynthesisStatus.FAILED)
