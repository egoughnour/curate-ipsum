"""
Graph-expanded RAG retrieval pipeline.

Pipeline: embed query → vector top-k → graph expansion → rerank → context pack
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


# -- Protocols for dependencies (avoid hard imports) --

class VectorStoreProtocol(Protocol):
    def search(self, embedding: np.ndarray, top_k: int, filters: Optional[dict]) -> list: ...

class EmbeddingProviderProtocol(Protocol):
    def embed(self, texts: list) -> np.ndarray: ...

class GraphStoreProtocol(Protocol):
    """Expected interface for the existing GraphStore."""
    def get_callers(self, node_id: str) -> List[str]: ...
    def get_callees(self, node_id: str) -> List[str]: ...
    def get_partition_members(self, partition_id: int) -> List[str]: ...
    def get_scc_members(self, node_id: str) -> List[str]: ...
    def get_node_text(self, node_id: str) -> str: ...
    def get_node_metadata(self, node_id: str) -> Dict[str, Any]: ...


@dataclass
class RAGResult:
    node_id: str
    file_path: str
    line_start: int
    line_end: int
    symbol_name: str
    symbol_kind: str  # "function", "class", "method"
    partition_id: int
    score: float
    expansion_source: str  # "vector", "caller", "callee", "partition", "scc"
    text: str


@dataclass
class RAGConfig:
    vector_top_k: int = 20
    expansion_hops: int = 1
    include_partition: bool = True
    include_scc: bool = True
    final_top_k: int = 10
    max_context_tokens: int = 8000


class RAGPipeline:
    """Graph-expanded retrieval-augmented generation pipeline."""

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        embedding_provider: EmbeddingProviderProtocol,
        graph_store: GraphStoreProtocol,
        config: Optional[RAGConfig] = None,
    ):
        self.vs = vector_store
        self.ep = embedding_provider
        self.gs = graph_store
        self.config = config or RAGConfig()

    def search(self, query: str, filters: Optional[dict] = None) -> List[RAGResult]:
        """Full pipeline: embed → vector search → graph expand → rerank → return."""
        # 1. Embed query
        query_vec = self.ep.embed([query])[0]

        # 2. Vector top-k
        hits = self.vs.search(query_vec, top_k=self.config.vector_top_k, filters=filters)

        # 3. Graph expansion
        expanded: Dict[str, RAGResult] = {}
        for hit in hits:
            nid = hit.id
            meta = hit.metadata
            self._add_result(expanded, nid, hit.score, "vector", meta)

            # Expand callers/callees
            for hop in range(self.config.expansion_hops):
                for caller_id in self.gs.get_callers(nid):
                    self._add_result(expanded, caller_id, hit.score * 0.8, "caller")
                for callee_id in self.gs.get_callees(nid):
                    self._add_result(expanded, callee_id, hit.score * 0.8, "callee")

            # Expand partition
            if self.config.include_partition:
                pid = meta.get("partition_id")
                if pid is not None:
                    for member_id in self.gs.get_partition_members(pid):
                        self._add_result(expanded, member_id, hit.score * 0.6, "partition")

            # Expand SCC
            if self.config.include_scc:
                for scc_id in self.gs.get_scc_members(nid):
                    self._add_result(expanded, scc_id, hit.score * 0.7, "scc")

        # 4. Rerank by score (could use cross-encoder here)
        results = sorted(expanded.values(), key=lambda r: r.score, reverse=True)
        return results[: self.config.final_top_k]

    def _add_result(
        self,
        results: Dict[str, RAGResult],
        node_id: str,
        score: float,
        source: str,
        metadata: Optional[dict] = None,
    ):
        if node_id in results:
            # Keep higher score
            if score > results[node_id].score:
                results[node_id].score = score
                results[node_id].expansion_source = source
            return

        try:
            meta = metadata or self.gs.get_node_metadata(node_id)
            text = self.gs.get_node_text(node_id)
        except Exception:
            return  # Skip nodes we can't resolve

        results[node_id] = RAGResult(
            node_id=node_id,
            file_path=meta.get("file_path", ""),
            line_start=meta.get("line_start", 0),
            line_end=meta.get("line_end", 0),
            symbol_name=meta.get("symbol_name", node_id),
            symbol_kind=meta.get("symbol_kind", "unknown"),
            partition_id=meta.get("partition_id", -1),
            score=score,
            expansion_source=source,
            text=text,
        )

    def pack_context(self, results: List[RAGResult]) -> str:
        """Pack results into a prompt-ready context string with provenance."""
        parts = []
        for r in results:
            header = f"# {r.symbol_name} ({r.symbol_kind}) — {r.file_path}:{r.line_start}-{r.line_end}"
            header += f"\n# partition={r.partition_id}, source={r.expansion_source}, score={r.score:.3f}"
            parts.append(f"{header}\n{r.text}")
        return "\n\n---\n\n".join(parts)
