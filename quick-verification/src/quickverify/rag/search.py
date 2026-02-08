"""Graph-expanded RAG retrieval pipeline.

Pipeline: embed query → vector top-k → graph expansion → rerank → context pack.

Graph expansion uses the existing GraphStore to pull structurally related code:
  - 1-hop callers/callees
  - Same Fiedler partition neighbors
  - Same SCC members
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

log = logging.getLogger(__name__)


# -- Protocols (avoid hard coupling) --

class VectorStoreProto(Protocol):
    def search(self, embedding: list, top_k: int, filters: Optional[dict]) -> list: ...

class EmbeddingProviderProto(Protocol):
    def embed(self, texts: list) -> list: ...

class GraphStoreProto(Protocol):
    """Expected interface for the existing GraphStore."""
    def get_callers(self, node_id: str) -> List[str]: ...
    def get_callees(self, node_id: str) -> List[str]: ...
    def get_partition_members(self, partition_id: int) -> List[str]: ...
    def get_scc_members(self, node_id: str) -> List[str]: ...
    def get_node_text(self, node_id: str) -> str: ...
    def get_node_metadata(self, node_id: str) -> Dict[str, Any]: ...


@dataclass
class RAGResult:
    """A single retrieval result with provenance."""
    node_id: str
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    symbol_name: str = ""
    symbol_kind: str = "unknown"
    partition_id: int = -1
    score: float = 0.0
    expansion_source: str = "vector"  # "vector"|"caller"|"callee"|"partition"|"scc"
    text: str = ""


@dataclass
class RAGConfig:
    """Tunable parameters for the RAG pipeline."""
    vector_top_k: int = 20
    expansion_hops: int = 1
    include_partition: bool = True
    include_scc: bool = True
    final_top_k: int = 10
    max_context_chars: int = 32_000
    caller_decay: float = 0.8
    callee_decay: float = 0.8
    partition_decay: float = 0.6
    scc_decay: float = 0.7


class RAGPipeline:
    """Graph-expanded retrieval-augmented generation pipeline.

    Usage::

        pipeline = RAGPipeline(vector_store, embedding_provider, graph_store)
        results = pipeline.search("does target_fn ever return negative?")
        context = pipeline.pack_context(results)
    """

    def __init__(
        self,
        vector_store: VectorStoreProto,
        embedding_provider: EmbeddingProviderProto,
        graph_store: GraphStoreProto,
        config: Optional[RAGConfig] = None,
    ):
        self.vs = vector_store
        self.ep = embedding_provider
        self.gs = graph_store
        self.config = config or RAGConfig()

    def search(self, query: str, filters: Optional[dict] = None) -> List[RAGResult]:
        """Run the full pipeline: embed → search → expand → rerank → return."""
        cfg = self.config

        # 1. Embed query
        query_vec = self.ep.embed([query])[0]

        # 2. Vector top-k
        hits = self.vs.search(query_vec, top_k=cfg.vector_top_k, filters=filters)

        # 3. Graph expansion
        expanded: Dict[str, RAGResult] = {}
        for hit in hits:
            nid = hit.id
            meta = hit.metadata
            self._add(expanded, nid, hit.score, "vector", meta)

            # Caller/callee expansion
            for _hop in range(cfg.expansion_hops):
                try:
                    for caller_id in self.gs.get_callers(nid):
                        self._add(expanded, caller_id, hit.score * cfg.caller_decay, "caller")
                except Exception:
                    pass
                try:
                    for callee_id in self.gs.get_callees(nid):
                        self._add(expanded, callee_id, hit.score * cfg.callee_decay, "callee")
                except Exception:
                    pass

            # Partition expansion
            if cfg.include_partition:
                pid = meta.get("partition_id") if isinstance(meta, dict) else None
                if pid is not None:
                    try:
                        for mid in self.gs.get_partition_members(pid):
                            self._add(expanded, mid, hit.score * cfg.partition_decay, "partition")
                    except Exception:
                        pass

            # SCC expansion
            if cfg.include_scc:
                try:
                    for sid in self.gs.get_scc_members(nid):
                        self._add(expanded, sid, hit.score * cfg.scc_decay, "scc")
                except Exception:
                    pass

        # 4. Rerank by score (descending)
        results = sorted(expanded.values(), key=lambda r: r.score, reverse=True)
        return results[: cfg.final_top_k]

    def pack_context(self, results: List[RAGResult]) -> str:
        """Assemble retrieval results into a prompt-ready context string."""
        parts = []
        total_chars = 0
        for r in results:
            header = f"# {r.symbol_name} ({r.symbol_kind}) — {r.file_path}:{r.line_start}-{r.line_end}"
            header += f"\n# partition={r.partition_id}, via={r.expansion_source}, score={r.score:.3f}"
            block = f"{header}\n{r.text}"
            if total_chars + len(block) > self.config.max_context_chars:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n\n---\n\n".join(parts)

    def _add(
        self,
        results: Dict[str, RAGResult],
        node_id: str,
        score: float,
        source: str,
        metadata: Optional[dict] = None,
    ):
        """Add or update a result, keeping the higher score."""
        if node_id in results:
            if score > results[node_id].score:
                results[node_id].score = score
                results[node_id].expansion_source = source
            return

        try:
            meta = metadata if isinstance(metadata, dict) else self.gs.get_node_metadata(node_id)
            text = self.gs.get_node_text(node_id)
        except Exception:
            return

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
