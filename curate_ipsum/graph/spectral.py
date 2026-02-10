"""
Graph-spectral analysis: Laplacian construction and Fiedler vector computation.

The Fiedler vector (eigenvector of the second-smallest eigenvalue of the
graph Laplacian) provides optimal bipartition of a connected graph. This
module builds sparse Laplacians from CallGraph objects and computes Fiedler
vectors for spectral partitioning.

References:
    Fiedler, M. (1973). Algebraic connectivity of graphs.
    Czechoslovak Mathematical Journal, 23(98), 298-305.

Requires: scipy>=1.10
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .models import CallGraph, EdgeKind


def _require_scipy() -> None:
    """Raise a clear error if scipy is not installed."""
    if not HAS_SCIPY:
        raise ImportError("scipy is required for spectral analysis. Install with: pip install 'curate-ipsum[graph]'")


def build_adjacency_matrix(
    graph: CallGraph,
    edge_kinds: set[EdgeKind] | None = None,
    weighted: bool = True,
) -> tuple["sp.csr_matrix", list[str]]:
    """
    Build a sparse adjacency matrix from a CallGraph.

    Args:
        graph: The call graph to convert.
        edge_kinds: Which edge types to include. Default: {CALLS} only.
        weighted: If True, use edge confidence as weight. If False, binary.

    Returns:
        (adjacency_matrix, node_id_list) where node_id_list[i] maps
        matrix row/col index i to a node ID string.
    """
    _require_scipy()

    if edge_kinds is None:
        edge_kinds = {EdgeKind.CALLS}

    # Stable ordering of nodes
    node_ids = sorted(graph.nodes.keys())
    node_index: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for edge in graph.edges:
        if edge.kind not in edge_kinds:
            continue
        src = node_index.get(edge.source_id)
        tgt = node_index.get(edge.target_id)
        if src is None or tgt is None:
            continue  # Skip edges to/from nodes not in graph

        weight = edge.confidence if weighted else 1.0
        rows.append(src)
        cols.append(tgt)
        data.append(weight)

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return adj, node_ids


def build_laplacian(
    graph: CallGraph,
    edge_kinds: set[EdgeKind] | None = None,
) -> tuple["sp.csr_matrix", list[str]]:
    """
    Build the symmetric graph Laplacian L = D - A_sym.

    For directed graphs, symmetrizes the adjacency matrix:
        A_sym = (A + A^T) / 2
    Then computes L = D - A_sym where D is the degree matrix.

    The Laplacian is always symmetric positive semi-definite with
    smallest eigenvalue 0 (corresponding to the constant eigenvector).

    Args:
        graph: The call graph.
        edge_kinds: Which edge types to include. Default: {CALLS}.

    Returns:
        (laplacian, node_id_list) — sparse Laplacian and node ID mapping.
    """
    _require_scipy()

    adj, node_ids = build_adjacency_matrix(graph, edge_kinds, weighted=True)
    n = adj.shape[0]

    if n == 0:
        return sp.csr_matrix((0, 0)), node_ids

    # Symmetrize: A_sym = (A + A^T) / 2
    a_sym = (adj + adj.T) / 2.0

    # Degree matrix: D_ii = sum of row i of A_sym
    degrees = np.asarray(a_sym.sum(axis=1)).flatten()
    D = sp.diags(degrees)

    # Laplacian: L = D - A_sym
    laplacian = D - a_sym

    return laplacian.tocsr(), node_ids


def find_connected_components(
    graph: CallGraph,
    edge_kinds: set[EdgeKind] | None = None,
) -> list[frozenset[str]]:
    """
    Find connected components of the graph (treating edges as undirected).

    Args:
        graph: The call graph.
        edge_kinds: Which edge types to consider for connectivity.

    Returns:
        List of frozensets, each containing the node IDs of one component.
    """
    if edge_kinds is None:
        edge_kinds = {EdgeKind.CALLS}

    # Build undirected adjacency list
    adj: dict[str, set[str]] = {nid: set() for nid in graph.nodes}
    for edge in graph.edges:
        if edge.kind not in edge_kinds:
            continue
        if edge.source_id in adj and edge.target_id in adj:
            adj[edge.source_id].add(edge.target_id)
            adj[edge.target_id].add(edge.source_id)

    visited: set[str] = set()
    components: list[frozenset[str]] = []

    for start in graph.nodes:
        if start in visited:
            continue
        # BFS
        component: set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adj.get(node, ()):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(frozenset(component))

    return components


@dataclass
class FiedlerResult:
    """Result of Fiedler vector computation for one connected component."""

    vector: np.ndarray
    """Fiedler vector values, one per node. vector[i] corresponds to node_ids[i]."""

    algebraic_connectivity: float
    """Second-smallest eigenvalue (λ₂). Measures how well-connected the graph is."""

    node_ids: list[str]
    """Mapping: vector[i] corresponds to node_ids[i]."""

    partition: dict[str, int]
    """Node ID → partition label (0 or 1) based on sign of Fiedler vector."""


def compute_fiedler(
    laplacian: "sp.csr_matrix",
    node_ids: list[str],
    tolerance: float = 1e-8,
) -> FiedlerResult:
    """
    Compute the Fiedler vector of a connected graph's Laplacian.

    The Fiedler vector is the eigenvector corresponding to λ₂ (the
    second-smallest eigenvalue). The sign of each component provides
    an optimal bipartition.

    Args:
        laplacian: Sparse symmetric Laplacian matrix (n×n).
        node_ids: Node ID strings, same order as matrix rows/cols.
        tolerance: Eigenvalue tolerance; λ₂ below this is treated as 0.

    Returns:
        FiedlerResult with vector, connectivity, and partition.

    Raises:
        ValueError: If graph appears disconnected (λ₂ ≈ 0).
    """
    _require_scipy()

    n = laplacian.shape[0]

    if n <= 1:
        vec = np.zeros(n)
        return FiedlerResult(
            vector=vec,
            algebraic_connectivity=0.0,
            node_ids=node_ids,
            partition=dict.fromkeys(node_ids, 0),
        )

    if n == 2:
        # Trivial case: two nodes → one in each partition
        vec = np.array([-1.0, 1.0])
        # λ₂ for two connected nodes = sum of edge weights * 2 / n
        diag = laplacian.diagonal()
        lam2 = float(min(diag)) if len(diag) > 0 else 0.0
        partition = {node_ids[0]: 0, node_ids[1]: 1}
        return FiedlerResult(
            vector=vec,
            algebraic_connectivity=lam2,
            node_ids=node_ids,
            partition=partition,
        )

    # For small graphs, use dense eigendecomposition (more reliable)
    if n <= 50:
        L_dense = laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        # Sort by eigenvalue (should already be sorted, but ensure)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        lam2 = float(eigenvalues[1])
        fiedler_vec = eigenvectors[:, 1]
    else:
        # Use sparse iterative solver for larger graphs
        try:
            eigenvalues, eigenvectors = spla.eigsh(
                laplacian.astype(np.float64),
                k=2,
                which="SM",
                tol=tolerance,
                maxiter=n * 100,
            )
            idx = np.argsort(eigenvalues)
            lam2 = float(eigenvalues[idx[1]])
            fiedler_vec = eigenvectors[:, idx[1]]
        except spla.ArpackNoConvergence:
            # Fallback to dense for convergence failures
            L_dense = laplacian.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            idx = np.argsort(eigenvalues)
            lam2 = float(eigenvalues[1])
            fiedler_vec = eigenvectors[:, 1]

    if lam2 < tolerance:
        raise ValueError(
            f"Graph appears disconnected (λ₂ = {lam2:.2e} < tolerance {tolerance:.2e}). "
            "Use compute_fiedler_components() for disconnected graphs."
        )

    # Build partition from sign of Fiedler vector
    partition: dict[str, int] = {}
    for i, nid in enumerate(node_ids):
        partition[nid] = 0 if fiedler_vec[i] < 0 else 1

    return FiedlerResult(
        vector=fiedler_vec,
        algebraic_connectivity=lam2,
        node_ids=node_ids,
        partition=partition,
    )


def compute_fiedler_components(
    graph: CallGraph,
    edge_kinds: set[EdgeKind] | None = None,
    tolerance: float = 1e-8,
) -> list[FiedlerResult]:
    """
    Compute Fiedler vectors per connected component.

    Handles disconnected graphs by decomposing into components first,
    then computing Fiedler for each component with ≥ 3 nodes.
    Components with 1-2 nodes get trivial partitions.

    Args:
        graph: The call graph (may be disconnected).
        edge_kinds: Which edge types to consider.
        tolerance: Eigenvalue tolerance.

    Returns:
        List of FiedlerResult, one per connected component.
    """
    _require_scipy()

    if edge_kinds is None:
        edge_kinds = {EdgeKind.CALLS}

    components = find_connected_components(graph, edge_kinds)
    results: list[FiedlerResult] = []

    for component in components:
        comp_ids = sorted(component)
        n = len(comp_ids)

        if n <= 2:
            # Trivial partition
            vec = np.zeros(n)
            if n == 2:
                vec[1] = 1.0
            partition = {nid: (0 if i == 0 else 1) for i, nid in enumerate(comp_ids)}
            results.append(
                FiedlerResult(
                    vector=vec,
                    algebraic_connectivity=0.0,
                    node_ids=comp_ids,
                    partition=partition,
                )
            )
            continue

        # Build subgraph for this component
        subgraph = _extract_subgraph(graph, component, edge_kinds)
        laplacian, sub_ids = build_laplacian(subgraph, edge_kinds)

        try:
            result = compute_fiedler(laplacian, sub_ids, tolerance)
            results.append(result)
        except ValueError:
            # Component is further disconnected at this edge_kinds filter;
            # treat as trivial
            vec = np.zeros(n)
            partition = dict.fromkeys(comp_ids, 0)
            results.append(
                FiedlerResult(
                    vector=vec,
                    algebraic_connectivity=0.0,
                    node_ids=comp_ids,
                    partition=partition,
                )
            )

    return results


def _extract_subgraph(
    graph: CallGraph,
    node_ids: frozenset[str],
    edge_kinds: set[EdgeKind] | None = None,
) -> CallGraph:
    """Extract a subgraph containing only the specified nodes."""
    subgraph = CallGraph()

    for nid in node_ids:
        node = graph.get_node(nid)
        if node is not None:
            subgraph.add_node(node)

    for edge in graph.edges:
        if edge_kinds and edge.kind not in edge_kinds:
            continue
        if edge.source_id in node_ids and edge.target_id in node_ids:
            subgraph.add_edge(edge)

    return subgraph
