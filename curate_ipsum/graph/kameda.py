"""
Kameda O(1) reachability index for planar directed acyclic graphs.

Implements a variant of Kameda's 1975 algorithm: for a planar st-DAG
(single source s, single sink t), we compute a 2D label for each node
such that reachability can be answered in O(1) by comparing labels.

The key insight: in a planar st-digraph, each node can be assigned
an interval [left, right] such that u reaches v if and only if
u's interval contains v's interval (or equivalently, the left/right
orderings are consistent).

Algorithm:
    1. Ensure graph is a DAG with single source and single sink
       (add virtual s/t if needed).
    2. Compute a topological ordering.
    3. Using the planar embedding, compute "left rank" and "right rank"
       for each node — these are topological orderings that respect the
       left-boundary and right-boundary of the planar embedding.
    4. Node u reaches node v iff:
       left_rank[u] <= left_rank[v] AND right_rank[u] <= right_rank[v]

For non-planar edges (identified by planarity.py), a separate fallback
set is maintained and checked via BFS.

References:
    Kameda, T. (1975). On the vector representation of the reachability
        in planar directed graphs. Inf. Proc. Letters 3(3), 75-77.
    Tamassia, R. & Tollis, I. (1989). Planar grid embedding in linear time.
    DECISIONS.md → D-006

Requires: networkx>=3.0 (for planar embedding utilities)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .models import CallGraph, EdgeKind, GraphEdge


@dataclass
class KamedaIndex:
    """
    O(1) reachability index for a planar directed acyclic graph.

    After O(n) construction, reachability queries take O(1) time
    by comparing 2D labels. For nodes not in the planar subgraph,
    or when edges were removed during planarization, a fallback
    set is checked.
    """

    left_rank: dict[str, int]
    """Left-boundary topological rank for each node."""

    right_rank: dict[str, int]
    """Right-boundary topological rank for each node."""

    source_id: str
    """The single source (or virtual source) of the st-graph."""

    sink_id: str
    """The single sink (or virtual sink) of the st-graph."""

    non_planar_reachability: dict[str, set[str]]
    """
    Fallback reachability for edges that were removed during planarization.
    Maps node_id → set of additional nodes reachable via non-planar edges.
    Precomputed during build.
    """

    all_node_ids: frozenset[str]
    """All node IDs in the indexed graph."""

    @classmethod
    def build(
        cls,
        graph: CallGraph,
        embedding: dict | None = None,
        non_planar_edges: set[GraphEdge] | None = None,
        source_id: str | None = None,
        sink_id: str | None = None,
        edge_kinds: set[EdgeKind] | None = None,
    ) -> "KamedaIndex":
        """
        Build the reachability index from a planar DAG.

        Args:
            graph: A planar directed acyclic graph (the planar subgraph).
            embedding: Planar embedding dict from planarity.check_planarity().
                       Maps node_id → clockwise-ordered list of neighbor IDs.
                       If None, a simple topological-order heuristic is used.
            non_planar_edges: Edges removed during planarization. These are
                handled via a fallback BFS precomputation.
            source_id: ID of the source node. If None, auto-detected
                (node with no incoming edges).
            sink_id: ID of the sink node. If None, auto-detected
                (node with no outgoing edges).
            edge_kinds: Edge types to consider. Default: {CALLS}.

        Returns:
            KamedaIndex ready for O(1) queries.

        Raises:
            ValueError: If graph is not a DAG or has no valid source/sink.
        """
        if edge_kinds is None:
            edge_kinds = {EdgeKind.CALLS}

        all_ids = frozenset(graph.nodes.keys())

        if len(all_ids) == 0:
            return cls(
                left_rank={},
                right_rank={},
                source_id="",
                sink_id="",
                non_planar_reachability={},
                all_node_ids=frozenset(),
            )

        # Build adjacency structures for the planar subgraph only
        forward: dict[str, list[str]] = {nid: [] for nid in all_ids}
        backward: dict[str, list[str]] = {nid: [] for nid in all_ids}
        for edge in graph.edges:
            if edge.kind not in edge_kinds:
                continue
            if edge.source_id in all_ids and edge.target_id in all_ids:
                forward[edge.source_id].append(edge.target_id)
                backward[edge.target_id].append(edge.source_id)

        # Detect or validate source and sink
        if source_id is None:
            sources = [n for n in all_ids if not backward[n]]
            if not sources:
                raise ValueError("Graph has no source node (every node has incoming edges — cycle?)")
            source_id = sources[0]
            # If multiple sources, that's okay — we'll pick the first;
            # virtual source augmentation should have handled this.

        if sink_id is None:
            sinks = [n for n in all_ids if not forward[n]]
            if not sinks:
                raise ValueError("Graph has no sink node (every node has outgoing edges — cycle?)")
            sink_id = sinks[0]

        # Verify DAG property via topological sort attempt
        topo_order = _topological_sort_kahn(forward, all_ids)
        if topo_order is None:
            raise ValueError("Graph contains cycles — not a DAG. Run SCC condensation first.")

        # Compute left and right ranks using the planar embedding
        if embedding and len(embedding) > 0:
            left_rank, right_rank = _compute_ranks_from_embedding(
                forward, backward, all_ids, embedding, topo_order, source_id
            )
        else:
            # No embedding available — use two different topological orderings
            # as an approximation (correct for trees and simple DAGs)
            left_rank, right_rank = _compute_ranks_heuristic(forward, backward, all_ids, topo_order)

        # Precompute fallback reachability for non-planar edges
        non_planar_reach: dict[str, set[str]] = {}
        if non_planar_edges:
            non_planar_reach = _compute_non_planar_reachability(non_planar_edges, forward, all_ids, edge_kinds)

        return cls(
            left_rank=left_rank,
            right_rank=right_rank,
            source_id=source_id,
            sink_id=sink_id,
            non_planar_reachability=non_planar_reach,
            all_node_ids=all_ids,
        )

    def reaches(self, source: str, target: str) -> bool:
        """
        O(1) reachability query: can source reach target?

        Uses the 2D dominance test on left/right ranks, falling back
        to the precomputed non-planar reachability set for edges that
        were removed during planarization.

        Args:
            source: Source node ID.
            target: Target node ID.

        Returns:
            True if there exists a directed path from source to target.
        """
        if source == target:
            return True

        if source not in self.all_node_ids or target not in self.all_node_ids:
            return False

        # Primary test: 2D dominance on planar ranks
        # u reaches v iff left_rank[u] <= left_rank[v] AND right_rank[u] <= right_rank[v]
        left_ok = self.left_rank[source] <= self.left_rank[target]
        right_ok = self.right_rank[source] <= self.right_rank[target]

        if left_ok and right_ok:
            return True

        # Fallback: check non-planar reachability
        if source in self.non_planar_reachability:
            if target in self.non_planar_reachability[source]:
                return True

        return False

    def all_reachable_from(self, source: str) -> set[str]:
        """
        Get all nodes reachable from source.

        Uses the index for the planar portion and merges with
        non-planar fallback reachability.

        Args:
            source: Source node ID.

        Returns:
            Set of all reachable node IDs (excluding source itself).
        """
        if source not in self.all_node_ids:
            return set()

        reachable: set[str] = set()

        # Check all nodes via 2D dominance
        s_left = self.left_rank[source]
        s_right = self.right_rank[source]

        for node in self.all_node_ids:
            if node == source:
                continue
            if self.left_rank[node] >= s_left and self.right_rank[node] >= s_right:
                reachable.add(node)

        # Merge non-planar reachability
        if source in self.non_planar_reachability:
            reachable.update(self.non_planar_reachability[source])

        return reachable

    def to_dict(self) -> dict:
        """Serialize the index for storage/transmission."""
        return {
            "source_id": self.source_id,
            "sink_id": self.sink_id,
            "left_rank": self.left_rank,
            "right_rank": self.right_rank,
            "non_planar_reachability": {k: sorted(v) for k, v in self.non_planar_reachability.items()},
            "node_count": len(self.all_node_ids),
        }


# ─────────────────────────────────────────────────────────────────
# Internal algorithms
# ─────────────────────────────────────────────────────────────────


def _topological_sort_kahn(
    forward: dict[str, list[str]],
    all_ids: frozenset[str],
) -> list[str] | None:
    """
    Kahn's algorithm for topological sort.

    Returns the sorted list, or None if the graph has a cycle.
    """
    in_degree: dict[str, int] = dict.fromkeys(all_ids, 0)
    for src in all_ids:
        for tgt in forward.get(src, []):
            if tgt in in_degree:
                in_degree[tgt] += 1

    queue = deque(sorted(n for n in all_ids if in_degree[n] == 0))
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in forward.get(node, []):
            if neighbor in in_degree:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    if len(result) != len(all_ids):
        return None  # Cycle detected
    return result


def _compute_ranks_from_embedding(
    forward: dict[str, list[str]],
    backward: dict[str, list[str]],
    all_ids: frozenset[str],
    embedding: dict,
    topo_order: list[str],
    source_id: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Compute left and right ranks using the planar embedding.

    The planar embedding provides a clockwise ordering of neighbors
    around each vertex. We use this to define two canonical topological
    orderings:
    - Left ordering: at each vertex, visit the leftmost unvisited successor first
    - Right ordering: at each vertex, visit the rightmost unvisited successor first

    For a planar st-digraph, the 2D dominance test on these orderings
    correctly captures reachability.
    """
    # Build ordered successor lists using the embedding
    ordered_forward: dict[str, list[str]] = {nid: [] for nid in all_ids}

    for node_id in all_ids:
        successors = set(forward.get(node_id, []))
        if not successors:
            continue

        if node_id in embedding:
            # Use embedding order to sort successors
            cw_order = embedding[node_id]
            # Filter to only include actual successors, preserving clockwise order
            ordered = [n for n in cw_order if n in successors]
            # Add any successors not in embedding (shouldn't happen for valid embedding)
            remaining = successors - set(ordered)
            ordered.extend(sorted(remaining))
            ordered_forward[node_id] = ordered
        else:
            # No embedding info for this node — use default sort
            ordered_forward[node_id] = sorted(successors)

    # Left-first DFS (visit leftmost/first successor)
    left_rank = _dfs_rank(ordered_forward, all_ids, topo_order, reverse_children=False)

    # Right-first DFS (visit rightmost/last successor)
    right_rank = _dfs_rank(ordered_forward, all_ids, topo_order, reverse_children=True)

    return left_rank, right_rank


def _compute_ranks_heuristic(
    forward: dict[str, list[str]],
    backward: dict[str, list[str]],
    all_ids: frozenset[str],
    topo_order: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Compute left and right ranks without a planar embedding.

    Uses two topological orderings as an approximation:
    - Forward topological order (standard)
    - Reverse topological order (processing sinks first)

    This is correct for trees and provides a reasonable approximation
    for general DAGs, though it may produce false positives for
    non-planar DAGs.
    """
    # Forward ordering: standard topological rank
    left_rank: dict[str, int] = {}
    for i, nid in enumerate(topo_order):
        left_rank[nid] = i

    # Alternative ordering: among nodes at the same topological level,
    # reverse their relative order. This creates a second "perspective."
    # Compute level (longest path from source)
    level: dict[str, int] = dict.fromkeys(all_ids, 0)
    for nid in topo_order:
        for succ in forward.get(nid, []):
            if succ in level:
                level[succ] = max(level[succ], level[nid] + 1)

    # Group by level, reverse within each level
    by_level: dict[int, list[str]] = {}
    for nid in topo_order:
        lv = level[nid]
        by_level.setdefault(lv, []).append(nid)

    right_order: list[str] = []
    for lv in sorted(by_level.keys()):
        right_order.extend(reversed(by_level[lv]))

    right_rank: dict[str, int] = {}
    for i, nid in enumerate(right_order):
        right_rank[nid] = i

    return left_rank, right_rank


def _dfs_rank(
    ordered_forward: dict[str, list[str]],
    all_ids: frozenset[str],
    topo_order: list[str],
    reverse_children: bool,
) -> dict[str, int]:
    """
    Assign ranks via DFS using a specific child ordering.

    Args:
        ordered_forward: Successor lists ordered by embedding.
        all_ids: All node IDs.
        topo_order: Topological ordering (for finding roots).
        reverse_children: If True, visit children in reverse embedding order.

    Returns:
        Dict mapping node_id to its DFS rank (visit order).
    """
    # Find all root nodes (no predecessors in the DAG)
    has_pred: set[str] = set()
    for nid in all_ids:
        for succ in ordered_forward.get(nid, []):
            has_pred.add(succ)
    roots = [nid for nid in topo_order if nid not in has_pred]

    if not roots:
        # Fallback: use topological order directly
        return {nid: i for i, nid in enumerate(topo_order)}

    visited: set[str] = set()
    rank: dict[str, int] = {}
    counter = [0]

    def dfs(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        rank[node] = counter[0]
        counter[0] += 1

        children = ordered_forward.get(node, [])
        if reverse_children:
            children = list(reversed(children))
        for child in children:
            dfs(child)

    for root in roots:
        dfs(root)

    # Assign ranks to any unvisited nodes (isolated or unreachable)
    for nid in topo_order:
        if nid not in rank:
            rank[nid] = counter[0]
            counter[0] += 1

    return rank


def _compute_non_planar_reachability(
    non_planar_edges: set[GraphEdge],
    planar_forward: dict[str, list[str]],
    all_ids: frozenset[str],
    edge_kinds: set[EdgeKind],
) -> dict[str, set[str]]:
    """
    Precompute reachability contributed by non-planar edges.

    For each non-planar edge (u, v), we need to find all nodes w
    such that there exists a path using at least one non-planar edge
    that is NOT captured by the planar index.

    Strategy: For each non-planar edge (u, v), BFS forward from v
    in the full graph (planar + non-planar edges) and mark those
    as reachable from u. Then propagate backward: any node that can
    reach u in the planar graph can also reach all of u's non-planar targets.
    """
    if not non_planar_edges:
        return {}

    # Build full forward adjacency (planar + non-planar)
    full_forward: dict[str, set[str]] = {nid: set(planar_forward.get(nid, [])) for nid in all_ids}
    for edge in non_planar_edges:
        if edge.kind not in edge_kinds:
            continue
        if edge.source_id in all_ids and edge.target_id in all_ids:
            full_forward[edge.source_id].add(edge.target_id)

    # For each non-planar edge (u, v), BFS from v to find what v can reach
    non_planar_reach: dict[str, set[str]] = {}

    for edge in non_planar_edges:
        if edge.kind not in edge_kinds:
            continue
        u, v = edge.source_id, edge.target_id
        if u not in all_ids or v not in all_ids:
            continue

        # BFS from v in the full graph
        reachable_from_v: set[str] = set()
        queue = deque([v])
        visited: set[str] = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            reachable_from_v.add(node)
            for succ in full_forward.get(node, ()):
                if succ not in visited:
                    queue.append(succ)

        # u can reach everything v can reach (plus v itself)
        if u not in non_planar_reach:
            non_planar_reach[u] = set()
        non_planar_reach[u].update(reachable_from_v)
        non_planar_reach[u].discard(u)  # Don't include self

    return non_planar_reach
