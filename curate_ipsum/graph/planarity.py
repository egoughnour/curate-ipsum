"""
Planar subgraph identification and Kuratowski subgraph extraction.

For each subgraph, determines if it is planar. If not, extracts the
Kuratowski subgraph (K₅ or K₃,₃ minor) and computes a maximal planar
subgraph by iteratively removing edges that break planarity.

Uses networkx for O(n) Boyer-Myrvold planarity testing.

References:
    Boyer, J. & Myrvold, W. (2004). On the cutting edge:
        Simplified O(n) planarity by edge addition.
    Kuratowski, K. (1930). Sur le problème des courbes gauches en topologie.
    DECISIONS.md → D-006

Requires: networkx>=3.0
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from .models import CallGraph, EdgeKind, GraphEdge, GraphNode


def _require_networkx() -> None:
    """Raise a clear error if networkx is not installed."""
    if not HAS_NETWORKX:
        raise ImportError(
            "networkx is required for planarity analysis. Install with: pip install 'curate-ipsum[graph]'"
        )


@dataclass
class PlanarityResult:
    """Result of planarity analysis on a subgraph."""

    is_planar: bool
    """Whether the graph is planar."""

    planar_subgraph: CallGraph
    """The maximal planar subgraph (equals the input graph if planar)."""

    non_planar_edges: set[GraphEdge]
    """Edges removed to achieve planarity (empty if already planar)."""

    kuratowski_edges: set[tuple[str, str]] | None
    """Edge set of the Kuratowski subgraph (K₅ or K₃,₃) if non-planar, else None."""

    embedding: dict | None
    """
    Planar embedding as a dict: node_id → ordered list of neighbor IDs
    representing the clockwise order of edges around each vertex.
    None if graph has 0 or 1 nodes.
    """


def callgraph_to_networkx(
    graph: CallGraph,
    edge_kinds: set[EdgeKind] | None = None,
    as_undirected: bool = False,
) -> "nx.DiGraph | nx.Graph":
    """
    Convert a CallGraph to a networkx graph.

    Preserves node and edge metadata as attributes.

    Args:
        graph: The CallGraph to convert.
        edge_kinds: Edge types to include. Default: all.
        as_undirected: If True, return an undirected Graph.

    Returns:
        networkx DiGraph (or Graph if as_undirected=True).
    """
    _require_networkx()

    G = nx.Graph() if as_undirected else nx.DiGraph()

    for node in graph.nodes.values():
        attrs = {
            "kind": node.kind.value,
            "name": node.name,
        }
        if node.location:
            attrs["file"] = node.location.file
            attrs["line_start"] = node.location.line_start
            attrs["line_end"] = node.location.line_end
        if node.metadata:
            attrs["metadata"] = dict(node.metadata)
        G.add_node(node.id, **attrs)

    for edge in graph.edges:
        if edge_kinds and edge.kind not in edge_kinds:
            continue
        attrs = {
            "kind": edge.kind.value,
            "confidence": edge.confidence,
            "is_conditional": edge.is_conditional,
            "is_dynamic": edge.is_dynamic,
        }
        G.add_edge(edge.source_id, edge.target_id, **attrs)

    return G


def networkx_to_callgraph(
    nx_graph: "nx.DiGraph | nx.Graph",
    original: CallGraph | None = None,
) -> CallGraph:
    """
    Convert a networkx graph back to a CallGraph.

    If an original CallGraph is provided, uses it to restore full
    GraphNode/GraphEdge metadata for nodes/edges present in nx_graph.

    Args:
        nx_graph: The networkx graph.
        original: Optional original CallGraph for metadata recovery.

    Returns:
        A new CallGraph.
    """
    _require_networkx()

    result = CallGraph()

    for node_id in nx_graph.nodes:
        if original and node_id in original.nodes:
            result.add_node(original.nodes[node_id])
        else:
            from .models import NodeKind

            attrs = nx_graph.nodes[node_id]
            kind_str = attrs.get("kind", "function")
            try:
                kind = NodeKind(kind_str)
            except ValueError:
                kind = NodeKind.FUNCTION
            result.add_node(
                GraphNode(
                    id=node_id,
                    kind=kind,
                    name=attrs.get("name", node_id),
                )
            )

    for u, v in nx_graph.edges:
        if original:
            # Try to find the matching edge in the original
            found = False
            for e in original.edges:
                if e.source_id == u and e.target_id == v:
                    result.add_edge(e)
                    found = True
                    break
            if not found:
                result.add_edge(
                    GraphEdge(
                        source_id=u,
                        target_id=v,
                        kind=EdgeKind.CALLS,
                    )
                )
        else:
            attrs = nx_graph.edges[u, v]
            kind_str = attrs.get("kind", "calls")
            try:
                kind = EdgeKind(kind_str)
            except ValueError:
                kind = EdgeKind.CALLS
            result.add_edge(
                GraphEdge(
                    source_id=u,
                    target_id=v,
                    kind=kind,
                    confidence=attrs.get("confidence", 1.0),
                )
            )

    return result


def check_planarity(
    graph: CallGraph,
    edge_kinds: set[EdgeKind] | None = None,
) -> PlanarityResult:
    """
    Test if a CallGraph is planar and extract planarity-related structures.

    Uses networkx's Boyer-Myrvold O(n) planarity test. If the graph is
    not planar, identifies a Kuratowski subgraph and computes a maximal
    planar subgraph by removing edges from the Kuratowski certificate.

    Planarity is tested on the underlying undirected graph (ignoring
    edge direction), since planarity is a property of the undirected
    structure.

    Args:
        graph: The CallGraph to test.
        edge_kinds: Edge types to consider. Default: {CALLS}.

    Returns:
        PlanarityResult with planarity status, planar subgraph,
        removed edges, and Kuratowski subgraph if applicable.
    """
    _require_networkx()

    if edge_kinds is None:
        edge_kinds = {EdgeKind.CALLS}

    # Convert to undirected networkx graph for planarity testing
    G_undirected = callgraph_to_networkx(graph, edge_kinds, as_undirected=True)
    n = G_undirected.number_of_nodes()

    if n <= 4:
        # Graphs with ≤ 4 nodes are always planar
        embedding_dict = None
        if n >= 2:
            is_planar, cert = nx.check_planarity(G_undirected)
            if is_planar:
                embedding_dict = _embedding_to_dict(cert)

        return PlanarityResult(
            is_planar=True,
            planar_subgraph=graph,
            non_planar_edges=set(),
            kuratowski_edges=None,
            embedding=embedding_dict,
        )

    is_planar, certificate = nx.check_planarity(G_undirected)

    if is_planar:
        # Graph is planar — certificate is the planar embedding
        embedding_dict = _embedding_to_dict(certificate)
        return PlanarityResult(
            is_planar=True,
            planar_subgraph=graph,
            non_planar_edges=set(),
            kuratowski_edges=None,
            embedding=embedding_dict,
        )

    # Non-planar — certificate is a Kuratowski subgraph
    kuratowski_edges = set(certificate.edges())

    # Compute maximal planar subgraph by iteratively removing
    # edges from the Kuratowski certificate until planar
    planar_graph, removed_edges = _compute_maximal_planar_subgraph(graph, G_undirected, kuratowski_edges, edge_kinds)

    # Get the embedding of the resulting planar subgraph
    G_planar_undirected = callgraph_to_networkx(planar_graph, edge_kinds, as_undirected=True)
    if G_planar_undirected.number_of_nodes() >= 2:
        is_p, emb = nx.check_planarity(G_planar_undirected)
        embedding_dict = _embedding_to_dict(emb) if is_p else None
    else:
        embedding_dict = None

    return PlanarityResult(
        is_planar=False,
        planar_subgraph=planar_graph,
        non_planar_edges=removed_edges,
        kuratowski_edges=kuratowski_edges,
        embedding=embedding_dict,
    )


def _embedding_to_dict(embedding: "nx.PlanarEmbedding") -> dict:
    """Convert a networkx PlanarEmbedding to a plain dict."""
    result: dict[str, list] = {}
    for node in embedding:
        neighbors = list(embedding.neighbors_cw_order(node))
        result[str(node)] = [str(n) for n in neighbors]
    return result


def _compute_maximal_planar_subgraph(
    original: CallGraph,
    G_undirected: "nx.Graph",
    kuratowski_edges: set[tuple],
    edge_kinds: set[EdgeKind],
) -> tuple[CallGraph, set[GraphEdge]]:
    """
    Compute a maximal planar subgraph by iteratively removing
    edges that participate in Kuratowski subgraphs.

    This is a heuristic — the true maximal planar subgraph problem
    is NP-hard. We remove one edge from each Kuratowski certificate
    until the graph becomes planar.

    Returns:
        (planar_callgraph, removed_edges)
    """
    G = G_undirected.copy()
    removed_undirected: set[tuple[str, str]] = set()

    max_iterations = G.number_of_edges()  # Safety bound
    iteration = 0

    while iteration < max_iterations:
        is_planar, certificate = nx.check_planarity(G)
        if is_planar:
            break

        # Remove one edge from the Kuratowski subgraph
        # Choose the edge with the highest betweenness in the certificate
        # (heuristic: removing high-betweenness edges is more likely to
        #  break the non-planar structure)
        cert_edges = list(certificate.edges())
        if not cert_edges:
            break

        # Simple heuristic: remove the first edge found in the certificate
        # that hasn't been removed yet
        edge_to_remove = cert_edges[0]
        G.remove_edge(*edge_to_remove)
        removed_undirected.add((str(edge_to_remove[0]), str(edge_to_remove[1])))

        iteration += 1

    # Build the planar CallGraph by excluding removed edges
    planar_graph = CallGraph()
    for node in original.nodes.values():
        planar_graph.add_node(node)

    removed_callgraph_edges: set[GraphEdge] = set()
    for edge in original.edges:
        if edge.kind not in edge_kinds:
            # Non-matching edge kinds pass through unchanged
            planar_graph.add_edge(edge)
            continue

        u, v = edge.source_id, edge.target_id
        if (u, v) in removed_undirected or (v, u) in removed_undirected:
            removed_callgraph_edges.add(edge)
        else:
            planar_graph.add_edge(edge)

    return planar_graph, removed_callgraph_edges
