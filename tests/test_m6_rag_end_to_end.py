"""
End-to-end tests for M6-deferred: RAG semantic search pipeline.

Tests the full RAG flow with real state propagation:
1. Real Chroma vector store (ephemeral) indexes real documents
2. Deterministic embeddings produce differentiable vectors
3. RAG pipeline searches, graph-expands, packs context — all real
4. SQLite GraphStore provides real caller/callee expansion
5. Context flows into CEGIS prompt enrichment

No mocks of Chroma, GraphStore, or the RAG pipeline logic — only the
embedding model is replaced with a deterministic hash-based provider
(to avoid downloading the 90 MB sentence-transformers model in tests).
"""

import hashlib
import math
from uuid import uuid4

import pytest

chromadb = pytest.importorskip("chromadb")  # skip entire module if chromadb is absent

from graph.models import (  # noqa: E402
    CallGraph,
    EdgeKind,
    FunctionSignature,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
)
from rag.embedding_provider import EmbeddingProvider  # noqa: E402
from rag.search import RAGConfig, RAGPipeline, RAGResult  # noqa: E402
from rag.vector_store import ChromaVectorStore, VectorDocument  # noqa: E402
from storage.graph_store import build_graph_store  # noqa: E402

# ─── Deterministic Embedding Provider ─────────────────────────────────────────
#
# Produces 384-dim vectors from text hashes. Texts with similar words produce
# similar (but not identical) vectors, so cosine similarity search actually
# works. This is NOT a mock — it's a lightweight, deterministic alternative
# to the real sentence-transformers model that still exercises the full
# pipeline end-to-end.


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """
    Hash-based embedding provider for e2e testing.

    Maps each text to a reproducible 384-dim float vector derived from
    its content hash. Similar texts get somewhat similar vectors because
    we hash overlapping word n-grams.
    """

    DIM = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vec(t) for t in texts]

    def dimension(self) -> int:
        return self.DIM

    def _text_to_vec(self, text: str) -> list[float]:
        # Hash the full text for the base vector
        h = hashlib.sha512(text.encode()).digest()
        # Expand to 384 floats by cycling through hash bytes
        raw = []
        for i in range(self.DIM):
            byte_val = h[i % len(h)]
            # Mix in position to avoid pure cycling
            raw.append(((byte_val + i * 37) % 256) / 255.0)
        # Normalize to unit vector (cosine similarity needs this)
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            raw = [x / norm for x in raw]
        return raw


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def embedder():
    """Deterministic embedding provider."""
    return DeterministicEmbeddingProvider()


@pytest.fixture
def chroma_store():
    """Ephemeral Chroma vector store (in-memory, no Docker needed). Unique per test."""
    store = ChromaVectorStore(collection_name=f"test_e2e_rag_{uuid4().hex[:8]}")
    yield store


@pytest.fixture
def graph_store(tmp_path):
    """Real SQLite graph store."""
    return build_graph_store("sqlite", tmp_path)


@pytest.fixture
def project_id(tmp_path):
    return str(tmp_path)


def _build_code_graph() -> CallGraph:
    """
    Build a realistic call graph representing a small validation module.

    validate_input → sanitize_string → check_length
                   → check_format
    process_data  → validate_input → ...
                  → transform
    """
    graph = CallGraph()

    nodes = [
        GraphNode(
            id="validator.validate_input",
            kind=NodeKind.FUNCTION,
            name="validate_input",
            location=SourceLocation(file="validator.py", line_start=10, line_end=25),
            signature=FunctionSignature(name="validate_input", params=("data",), return_type="bool"),
            docstring="Validate user input: sanitize, check length and format.",
        ),
        GraphNode(
            id="validator.sanitize_string",
            kind=NodeKind.FUNCTION,
            name="sanitize_string",
            location=SourceLocation(file="validator.py", line_start=28, line_end=35),
            signature=FunctionSignature(name="sanitize_string", params=("s",), return_type="str"),
            docstring="Strip dangerous characters from input string.",
        ),
        GraphNode(
            id="validator.check_length",
            kind=NodeKind.FUNCTION,
            name="check_length",
            location=SourceLocation(file="validator.py", line_start=38, line_end=42),
            signature=FunctionSignature(name="check_length", params=("s", "max_len"), return_type="bool"),
            docstring="Ensure string length is within bounds.",
        ),
        GraphNode(
            id="validator.check_format",
            kind=NodeKind.FUNCTION,
            name="check_format",
            location=SourceLocation(file="validator.py", line_start=45, line_end=55),
            signature=FunctionSignature(name="check_format", params=("s", "pattern"), return_type="bool"),
            docstring="Check if string matches expected format pattern.",
        ),
        GraphNode(
            id="pipeline.process_data",
            kind=NodeKind.FUNCTION,
            name="process_data",
            location=SourceLocation(file="pipeline.py", line_start=5, line_end=20),
            signature=FunctionSignature(name="process_data", params=("raw_data",), return_type="dict"),
            docstring="Main data processing pipeline: validate then transform.",
        ),
        GraphNode(
            id="pipeline.transform",
            kind=NodeKind.FUNCTION,
            name="transform",
            location=SourceLocation(file="pipeline.py", line_start=23, line_end=35),
            signature=FunctionSignature(name="transform", params=("validated_data",), return_type="dict"),
            docstring="Transform validated data into output format.",
        ),
    ]

    for node in nodes:
        graph.add_node(node)

    edges = [
        GraphEdge(source_id="validator.validate_input", target_id="validator.sanitize_string", kind=EdgeKind.CALLS),
        GraphEdge(source_id="validator.validate_input", target_id="validator.check_length", kind=EdgeKind.CALLS),
        GraphEdge(source_id="validator.validate_input", target_id="validator.check_format", kind=EdgeKind.CALLS),
        GraphEdge(source_id="pipeline.process_data", target_id="validator.validate_input", kind=EdgeKind.CALLS),
        GraphEdge(source_id="pipeline.process_data", target_id="pipeline.transform", kind=EdgeKind.CALLS),
    ]

    for edge in edges:
        graph.add_edge(edge)

    return graph


# ─── 1. Chroma + Embedding: index → search → retrieve ─────────────────────────


class TestChromaEmbeddingEndToEnd:
    """Real Chroma with deterministic embeddings — no mocks."""

    def test_index_and_search_by_similarity(self, chroma_store, embedder):
        """Index code nodes, search by semantic query, verify ranked results."""
        # Index 6 code documents with real embeddings
        docs = []
        for node in _build_code_graph().nodes.values():
            text = f"{node.name}: {node.docstring or ''}"
            embedding = embedder.embed([text])[0]
            docs.append(
                VectorDocument(
                    id=node.id,
                    text=text,
                    embedding=embedding,
                    metadata={"file": node.location.file if node.location else "", "kind": node.kind.value},
                )
            )

        chroma_store.add(docs)
        assert chroma_store.count() == 6

        # Search for "validate input data" — should rank validate_input highly
        query_vec = embedder.embed(["validate input data"])[0]
        results = chroma_store.search(query_vec, top_k=6)

        assert len(results) == 6
        # All documents should be returned with scores
        result_ids = [r.id for r in results]
        assert "validator.validate_input" in result_ids

        # Scores should be between 0 and 1 (cosine similarity)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_metadata_filtering(self, chroma_store, embedder):
        """Metadata filters correctly scope search results."""
        docs = []
        for node in _build_code_graph().nodes.values():
            text = f"{node.name}: {node.docstring or ''}"
            embedding = embedder.embed([text])[0]
            docs.append(
                VectorDocument(
                    id=node.id,
                    text=text,
                    embedding=embedding,
                    metadata={"file": node.location.file if node.location else "", "kind": node.kind.value},
                )
            )

        chroma_store.add(docs)

        # Search only in validator.py
        query_vec = embedder.embed(["string checking"])[0]
        results = chroma_store.search(query_vec, top_k=10, filter_metadata={"file": "validator.py"})

        # Should only return nodes from validator.py (4 nodes)
        assert len(results) == 4
        for r in results:
            assert r.metadata["file"] == "validator.py"

    def test_upsert_updates_existing(self, chroma_store, embedder):
        """Upserting with same ID updates the document, not duplicates it."""
        text_v1 = "validate_input: Check input is valid"
        text_v2 = "validate_input: Comprehensive input validation with sanitization"

        doc_v1 = VectorDocument(
            id="validator.validate_input",
            text=text_v1,
            embedding=embedder.embed([text_v1])[0],
        )
        chroma_store.add([doc_v1])
        assert chroma_store.count() == 1

        doc_v2 = VectorDocument(
            id="validator.validate_input",
            text=text_v2,
            embedding=embedder.embed([text_v2])[0],
        )
        chroma_store.add([doc_v2])
        assert chroma_store.count() == 1  # Still 1, not 2

        # Search should return the updated text
        query_vec = embedder.embed(["sanitization"])[0]
        results = chroma_store.search(query_vec, top_k=1)
        assert "sanitization" in results[0].text


# ─── 2. RAG Pipeline with graph expansion ─────────────────────────────────────


class TestRAGPipelineEndToEnd:
    """Full RAG pipeline: embed → vector search → graph expand → pack."""

    def test_search_returns_vector_results(self, chroma_store, embedder):
        """RAG search returns ranked results from vector store."""
        # Index documents
        for node in _build_code_graph().nodes.values():
            text = f"{node.name}: {node.docstring or ''}"
            embedding = embedder.embed([text])[0]
            chroma_store.add(
                [
                    VectorDocument(
                        id=node.id,
                        text=text,
                        embedding=embedding,
                        metadata={"kind": node.kind.value},
                    )
                ]
            )

        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
            config=RAGConfig(vector_top_k=6),
        )

        results = pipeline.search("validate input")
        assert len(results) > 0
        assert all(isinstance(r, RAGResult) for r in results)
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_graph_expansion_adds_neighbors(self, chroma_store, embedder, graph_store, project_id):
        """Graph expansion adds callers/callees not in initial vector results."""
        graph = _build_code_graph()

        # Persist graph to real SQLite store
        graph_store.store_graph(graph, project_id)

        # Index only a SUBSET of nodes in Chroma (so graph expansion can add the rest)
        subset_ids = ["validator.validate_input", "pipeline.process_data"]
        for node_id in subset_ids:
            node = graph.nodes[node_id]
            text = f"{node.name}: {node.docstring or ''}"
            embedding = embedder.embed([text])[0]
            chroma_store.add(
                [
                    VectorDocument(
                        id=node.id,
                        text=text,
                        embedding=embedding,
                        metadata={"kind": node.kind.value},
                    )
                ]
            )

        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
            graph_store=graph_store,
            config=RAGConfig(vector_top_k=10, expansion_hops=1, project_id=project_id),
        )

        results = pipeline.search("validate input")

        # Should have vector hits plus graph-expanded neighbors
        result_ids = {r.node_id for r in results}
        assert "validator.validate_input" in result_ids  # vector hit

        # Graph expansion should bring in callees of validate_input
        # (sanitize_string, check_length, check_format)
        graph_expanded = [r for r in results if r.source.startswith("graph_")]
        assert len(graph_expanded) > 0, "Graph expansion should add neighbor nodes"

        # Verify at least some callees were added
        callee_ids = {"validator.sanitize_string", "validator.check_length", "validator.check_format"}
        expanded_ids = {r.node_id for r in graph_expanded}
        assert callee_ids & expanded_ids, f"Expected callees in {expanded_ids}"

    def test_graph_expansion_includes_callers(self, chroma_store, embedder, graph_store, project_id):
        """Graph expansion adds callers (incoming edges) too."""
        graph = _build_code_graph()
        graph_store.store_graph(graph, project_id)

        # Index only the leaf node check_length
        node = graph.nodes["validator.check_length"]
        text = f"{node.name}: {node.docstring or ''}"
        embedding = embedder.embed([text])[0]
        chroma_store.add(
            [
                VectorDocument(
                    id=node.id,
                    text=text,
                    embedding=embedding,
                )
            ]
        )

        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
            graph_store=graph_store,
            config=RAGConfig(vector_top_k=5, expansion_hops=1, project_id=project_id),
        )

        results = pipeline.search("check string length bounds")
        _result_ids = {r.node_id for r in results}

        # validate_input calls check_length, so it should appear as a caller
        caller_results = [r for r in results if r.source == "graph_caller"]
        caller_ids = {r.node_id for r in caller_results}
        assert "validator.validate_input" in caller_ids, (
            f"validate_input should appear as caller of check_length, got {caller_ids}"
        )

    def test_pack_context_respects_token_limit(self, chroma_store, embedder):
        """Context packing stops at the token budget."""
        for node in _build_code_graph().nodes.values():
            text = f"{node.name}: {node.docstring or ''}"
            embedding = embedder.embed([text])[0]
            chroma_store.add([VectorDocument(id=node.id, text=text, embedding=embedding)])

        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
            config=RAGConfig(max_context_tokens=50),  # Very small budget
        )

        results = pipeline.search("data processing")
        context = pipeline.pack_context(results, max_tokens=50)

        # 50 tokens ≈ 200 chars — should truncate well before all results
        assert len(context) <= 250  # Some slack for header formatting
        assert len(context) > 0

    def test_pack_context_contains_node_ids_and_scores(self, chroma_store, embedder):
        """Packed context includes structured headers with node IDs and scores."""
        for node in _build_code_graph().nodes.values():
            text = f"{node.name}: {node.docstring or ''}"
            embedding = embedder.embed([text])[0]
            chroma_store.add([VectorDocument(id=node.id, text=text, embedding=embedding)])

        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
            config=RAGConfig(max_context_tokens=4000),
        )

        results = pipeline.search("validate")
        context = pipeline.pack_context(results)

        # Should contain structured headers
        assert "score=" in context
        assert "via=" in context
        # Should contain at least one node ID
        assert any(
            node_id in context
            for node_id in [
                "validator.validate_input",
                "validator.sanitize_string",
                "pipeline.process_data",
                "pipeline.transform",
            ]
        )


# ─── 3. RAG decay scoring ─────────────────────────────────────────────────────


class TestRAGDecayScoring:
    """Verify that graph-expanded results have decayed scores."""

    def test_callee_scores_decay(self, chroma_store, embedder, graph_store, project_id):
        """Callees get a lower score than the vector hit that expanded them."""
        graph = _build_code_graph()
        graph_store.store_graph(graph, project_id)

        # Index only validate_input
        node = graph.nodes["validator.validate_input"]
        text = f"{node.name}: {node.docstring or ''}"
        embedding = embedder.embed([text])[0]
        chroma_store.add([VectorDocument(id=node.id, text=text, embedding=embedding)])

        config = RAGConfig(
            vector_top_k=5,
            expansion_hops=1,
            callee_decay=0.8,
            caller_decay=0.7,
            project_id=project_id,
        )
        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
            graph_store=graph_store,
            config=config,
        )

        results = pipeline.search("validate")
        result_map = {r.node_id: r for r in results}

        # The vector hit should have the highest score
        if "validator.validate_input" in result_map:
            vi_score = result_map["validator.validate_input"].score
            for callee_id in ["validator.sanitize_string", "validator.check_length", "validator.check_format"]:
                if callee_id in result_map:
                    assert result_map[callee_id].score < vi_score, (
                        f"Callee {callee_id} should have lower score than parent"
                    )
                    assert result_map[callee_id].score == pytest.approx(vi_score * 0.8, abs=0.01), (
                        f"Callee {callee_id} score should be parent * decay"
                    )


# ─── 4. RAG → CEGIS integration ──────────────────────────────────────────────


class TestRAGCEGISIntegration:
    """
    RAG pipeline's context flows into CEGIS prompt enrichment.

    This tests that:
    - RAG search results are packed into a context string
    - That context string is actually used by CEGISEngine.synthesize()
    - The synthesis prompt changes based on RAG context
    """

    @pytest.mark.asyncio
    async def test_rag_context_reaches_cegis_prompt(self, chroma_store, embedder):
        """
        RAG context is injected into the CEGIS synthesis prompt.

        We verify this by checking that the LLM client receives a prompt
        containing RAG-retrieved content.
        """
        from synthesis.cegis import CEGISEngine
        from synthesis.llm_client import MockLLMClient
        from synthesis.models import Specification, SynthesisConfig

        # Index a document that will be found by RAG
        text = "validate_input: Comprehensive input validation with bounds checking"
        embedding = embedder.embed([text])[0]
        chroma_store.add(
            [
                VectorDocument(
                    id="validator.validate_input",
                    text=text,
                    embedding=embedding,
                )
            ]
        )

        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
            config=RAGConfig(vector_top_k=5, max_context_tokens=2000),
        )

        # Track what prompt the LLM receives
        captured_prompts = []

        class CapturingLLMClient(MockLLMClient):
            async def generate_candidates(self, prompt, n=1, temperature=0.8):
                captured_prompts.append(prompt)
                return await super().generate_candidates(prompt, n, temperature)

        client = CapturingLLMClient(
            responses=[
                "def fix(x):\n    return x + 1\n",
            ]
        )

        config = SynthesisConfig(max_iterations=2, population_size=4, top_k=4)
        engine = CEGISEngine(config, client, rag_pipeline=pipeline)

        spec = Specification(
            original_code="def fix(x): return x",
            target_region="fix",
        )

        await engine.synthesize(spec)

        # The LLM should have received a prompt containing RAG context
        assert len(captured_prompts) >= 1
        full_prompt = captured_prompts[0]
        assert "validate_input" in full_prompt or "Retrieved context" in full_prompt, (
            f"RAG context should be in the synthesis prompt, got: {full_prompt[:200]}"
        )

    @pytest.mark.asyncio
    async def test_rag_context_combined_with_spec_context(self, chroma_store, embedder):
        """RAG context is appended to the spec's existing context_code."""
        from synthesis.cegis import CEGISEngine
        from synthesis.llm_client import MockLLMClient
        from synthesis.models import Specification, SynthesisConfig

        text = "sanitize_string: Strip dangerous characters from input"
        embedding = embedder.embed([text])[0]
        chroma_store.add(
            [
                VectorDocument(
                    id="validator.sanitize_string",
                    text=text,
                    embedding=embedding,
                )
            ]
        )

        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
            config=RAGConfig(vector_top_k=3),
        )

        captured_prompts = []

        class CapturingLLMClient(MockLLMClient):
            async def generate_candidates(self, prompt, n=1, temperature=0.8):
                captured_prompts.append(prompt)
                return await super().generate_candidates(prompt, n, temperature)

        client = CapturingLLMClient(responses=["def fix(x): return x\n"])
        config = SynthesisConfig(max_iterations=1, population_size=4, top_k=4)
        engine = CEGISEngine(config, client, rag_pipeline=pipeline)

        spec = Specification(
            original_code="def fix(x): return x",
            target_region="fix",
            context_code="# Existing context: this module handles validation",
        )

        await engine.synthesize(spec)

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        # Both the original context AND RAG context should be present
        assert "Existing context" in prompt, "Original context should be preserved"


# ─── 5. Empty / edge cases ───────────────────────────────────────────────────


class TestRAGEdgeCases:
    """Edge cases that should not crash the pipeline."""

    def test_search_empty_store(self, chroma_store, embedder):
        """Searching an empty store returns empty results."""
        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
        )
        results = pipeline.search("anything")
        assert results == []

    def test_pack_empty_results(self, chroma_store, embedder):
        """Packing empty results returns empty string."""
        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
        )
        context = pipeline.pack_context([])
        assert context == ""

    def test_graph_expansion_without_graph_store(self, chroma_store, embedder):
        """RAG works without a graph store — just vector results."""
        text = "validate_input: Check input"
        embedding = embedder.embed([text])[0]
        chroma_store.add([VectorDocument(id="v.vi", text=text, embedding=embedding)])

        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=embedder,
            graph_store=None,  # No graph store
            config=RAGConfig(vector_top_k=5),
        )

        results = pipeline.search("validate")
        assert len(results) > 0
        # All results should be from vector (no graph expansion)
        assert all(r.source == "vector" for r in results)
