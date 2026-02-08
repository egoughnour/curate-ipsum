"""Tests for the RAG pipeline — graph-expanded retrieval."""
import pytest

from quickverify.rag.search import RAGConfig, RAGPipeline, RAGResult
from quickverify.rag.vector_store import SearchResult


class TestRAGPipeline:
    def test_search_returns_results(self, mock_vector_store, mock_embedding_provider, mock_graph_store):
        # Seed the vector store
        mock_vector_store.add(
            ids=["fn_a", "fn_b"],
            embeddings=[[0.1] * 384, [0.2] * 384],
            metadata=[
                {"file_path": "src/a.py", "symbol_name": "fn_a", "symbol_kind": "function", "partition_id": 0,
                 "line_start": 10, "line_end": 20},
                {"file_path": "src/b.py", "symbol_name": "fn_b", "symbol_kind": "function", "partition_id": 0,
                 "line_start": 5, "line_end": 15},
            ],
        )

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
            graph_store=mock_graph_store,
            config=RAGConfig(vector_top_k=5, final_top_k=10),
        )

        results = pipeline.search("does fn_a call fn_b?")
        assert len(results) > 0
        # Should include expansion results (callees of fn_a, etc.)
        node_ids = {r.node_id for r in results}
        assert "fn_a" in node_ids

    def test_pack_context(self, mock_vector_store, mock_embedding_provider, mock_graph_store):
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
            graph_store=mock_graph_store,
        )
        results = [
            RAGResult(node_id="fn_a", file_path="a.py", symbol_name="fn_a",
                      symbol_kind="function", score=0.9, text="def fn_a(): pass"),
        ]
        ctx = pipeline.pack_context(results)
        assert "fn_a" in ctx
        assert "function" in ctx

    def test_empty_store_returns_empty(self, mock_vector_store, mock_embedding_provider, mock_graph_store):
        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
            graph_store=mock_graph_store,
        )
        results = pipeline.search("anything")
        assert results == []
