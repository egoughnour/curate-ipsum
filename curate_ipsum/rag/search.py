"""
RAG search pipeline with graph-expanded retrieval.

Vector top-k → graph expansion (callers/callees via GraphStore) →
rerank by combined score → pack into LLM context.

Decision: D-017
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from curate_ipsum.rag.embedding_provider import EmbeddingProvider
from curate_ipsum.rag.vector_store import VectorSearchResult, VectorStore

if TYPE_CHECKING:
    from curate_ipsum.storage.graph_store import GraphStore

LOG = logging.getLogger("rag.search")


@dataclass
class RAGConfig:
    """Configuration for the RAG search pipeline."""

    vector_top_k: int = 20
    expansion_hops: int = 1
    caller_decay: float = 0.7
    callee_decay: float = 0.8
    max_context_tokens: int = 4000
    project_id: str = "default"


@dataclass
class RAGResult:
    """A single result from the RAG pipeline."""

    node_id: str
    text: str
    score: float
    source: str = "vector"  # "vector", "graph_caller", "graph_callee"
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """
    Code-aware retrieval pipeline.

    Combines vector similarity search with graph-based expansion using
    the project's existing GraphStore (D-014) for caller/callee relationships.

    Usage::

        pipeline = RAGPipeline(
            vector_store=chroma_store,
            embedding_provider=local_embedder,
            graph_store=sqlite_graph_store,  # optional
        )
        results = pipeline.search("function that validates input")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        graph_store: "GraphStore" | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self._vs = vector_store
        self._embed = embedding_provider
        self._gs = graph_store
        self._config = config or RAGConfig()

    def search(self, query: str) -> list[RAGResult]:
        """
        Search for code relevant to the query.

        1. Embed query
        2. Vector search for top-k
        3. Graph-expand results (callers + callees)
        4. Deduplicate and rerank
        """
        config = self._config

        # Step 1: Embed query
        embeddings = self._embed.embed([query])
        if not embeddings:
            return []
        query_vec = embeddings[0]

        # Step 2: Vector search
        vector_results = self._vs.search(query_vec, top_k=config.vector_top_k)

        # Convert to RAGResults
        results: dict[str, RAGResult] = {}
        for vr in vector_results:
            results[vr.id] = RAGResult(
                node_id=vr.id,
                text=vr.text,
                score=vr.score,
                source="vector",
                metadata=vr.metadata,
            )

        # Step 3: Graph expansion (if GraphStore available)
        if self._gs and vector_results:
            self._expand_graph(results, vector_results, config)

        # Step 4: Sort by score descending
        ranked = sorted(results.values(), key=lambda r: r.score, reverse=True)
        return ranked

    def _expand_graph(
        self,
        results: dict[str, RAGResult],
        vector_results: list[VectorSearchResult],
        config: RAGConfig,
    ) -> None:
        """Expand vector results using graph neighborhood."""
        for vr in vector_results[:10]:  # Expand top 10 vector hits
            node_id = vr.id

            for hop in range(config.expansion_hops):
                decay = config.callee_decay ** (hop + 1)

                # Callees (outgoing)
                try:
                    callees = self._gs.get_neighbors(node_id, config.project_id, direction="outgoing")
                    for callee_id in callees:
                        if callee_id not in results:
                            # Try to get node data for text
                            node_data = self._gs.get_node(callee_id, config.project_id)
                            text = ""
                            if node_data:
                                text = node_data.get("label", node_data.get("id", callee_id))
                            results[callee_id] = RAGResult(
                                node_id=callee_id,
                                text=text,
                                score=vr.score * decay,
                                source="graph_callee",
                                metadata=node_data or {},
                            )
                except Exception as exc:
                    LOG.debug("Graph expansion (callees) failed for %s: %s", node_id, exc)

                # Callers (incoming)
                caller_decay = config.caller_decay ** (hop + 1)
                try:
                    callers = self._gs.get_neighbors(node_id, config.project_id, direction="incoming")
                    for caller_id in callers:
                        if caller_id not in results:
                            node_data = self._gs.get_node(caller_id, config.project_id)
                            text = ""
                            if node_data:
                                text = node_data.get("label", node_data.get("id", caller_id))
                            results[caller_id] = RAGResult(
                                node_id=caller_id,
                                text=text,
                                score=vr.score * caller_decay,
                                source="graph_caller",
                                metadata=node_data or {},
                            )
                except Exception as exc:
                    LOG.debug("Graph expansion (callers) failed for %s: %s", node_id, exc)

    def pack_context(self, results: list[RAGResult], max_tokens: int | None = None) -> str:
        """
        Pack RAG results into a single context string for LLM prompt injection.

        Respects the token budget (estimated at 4 chars per token).
        """
        limit = max_tokens or self._config.max_context_tokens
        char_limit = limit * 4  # rough chars-per-token estimate

        parts: list[str] = []
        total_chars = 0

        for r in results:
            entry = f"## {r.node_id} (score={r.score:.2f}, via={r.source})\n{r.text}\n"
            if total_chars + len(entry) > char_limit:
                break
            parts.append(entry)
            total_chars += len(entry)

        return "\n".join(parts)
