"""
Recursive Fiedler partitioning with virtual sink/source augmentation.

Recursively applies spectral bipartition (Fiedler vector sign) to
produce a binary partition tree. Each leaf is a small, well-connected
subgraph. After partitioning, each partition can be augmented with
virtual source/sink nodes for uniform reachability queries.

References:
    architectural_vision.md § A.2-A.4
    DECISIONS.md → D-005, D-008
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    CallGraph,
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
)
from .spectral import (
    _extract_subgraph,
    build_laplacian,
    compute_fiedler,
    find_connected_components,
)


@dataclass
class Partition:
    """
    A node in the partition tree.

    Leaf partitions represent actual groups of code nodes.
    Internal partitions have two children from Fiedler bipartition.
    """

    id: str
    """Partition identifier. Binary tree scheme: root='0', left='0.0', right='0.1'."""

    node_ids: frozenset[str]
    """Original graph node IDs belonging to this partition (and all descendants)."""

    children: tuple["Partition", "Partition"] | None = None
    """Left and right child partitions, or None for leaf nodes."""

    fiedler_value: float | None = None
    """Algebraic connectivity (λ₂) of this subgraph. Higher = more connected."""

    depth: int = 0
    """Depth in the partition tree. Root = 0."""

    @property
    def is_leaf(self) -> bool:
        """True if this is a leaf partition (no children)."""
        return self.children is None

    @property
    def size(self) -> int:
        """Number of nodes in this partition."""
        return len(self.node_ids)


class GraphPartitioner:
    """
    Recursive Fiedler partitioner.

    Recursively bipartitions a CallGraph using the sign of the Fiedler
    vector, producing a binary tree of progressively finer partitions.

    Recursion stops when any of these conditions are met:
    - Partition size < min_partition_size
    - Tree depth >= max_depth
    - Algebraic connectivity λ₂ > connectivity_threshold
    """

    def __init__(
        self,
        min_partition_size: int = 3,
        max_depth: int = 20,
        connectivity_threshold: float = 0.0,
        edge_kinds: set[EdgeKind] | None = None,
    ):
        """
        Args:
            min_partition_size: Don't split partitions smaller than this.
            max_depth: Maximum recursion depth.
            connectivity_threshold: Stop splitting if λ₂ exceeds this
                (0.0 = always split until size limit).
            edge_kinds: Edge types to include in partitioning. Default: {CALLS}.
        """
        self.min_partition_size = max(2, min_partition_size)
        self.max_depth = max_depth
        self.connectivity_threshold = connectivity_threshold
        self.edge_kinds = edge_kinds or {EdgeKind.CALLS}

    def partition(self, graph: CallGraph) -> Partition:
        """
        Recursively bipartition the graph.

        Handles disconnected graphs by first decomposing into components,
        then partitioning each component independently.

        Args:
            graph: The call graph to partition.

        Returns:
            Root Partition node of the partition tree.
        """
        all_ids = frozenset(graph.nodes.keys())
        if len(all_ids) == 0:
            return Partition(id="0", node_ids=frozenset(), depth=0)

        # Handle disconnected components
        components = find_connected_components(graph, self.edge_kinds)

        if len(components) == 1:
            # Single connected component — partition directly
            return self._partition_recursive(graph, all_ids, "0", 0)
        else:
            # Multiple components — each gets its own subtree
            children_partitions: list[Partition] = []
            for i, comp in enumerate(sorted(components, key=len, reverse=True)):
                sub = _extract_subgraph(graph, comp, self.edge_kinds)
                child = self._partition_recursive(sub, comp, f"0.{i}", 1)
                children_partitions.append(child)

            # If exactly 2 components, make them left/right children
            if len(children_partitions) == 2:
                return Partition(
                    id="0",
                    node_ids=all_ids,
                    children=(children_partitions[0], children_partitions[1]),
                    depth=0,
                )

            # More than 2 components — build a balanced tree
            root = self._build_balanced_tree(children_partitions, "0", 0)
            return root

    def _partition_recursive(
        self,
        graph: CallGraph,
        node_ids: frozenset[str],
        partition_id: str,
        depth: int,
    ) -> Partition:
        """Recursively partition a connected subgraph."""
        n = len(node_ids)

        # Base cases: stop recursion
        if n < self.min_partition_size or depth >= self.max_depth:
            return Partition(
                id=partition_id,
                node_ids=node_ids,
                depth=depth,
            )

        # Build subgraph and compute Fiedler
        subgraph = _extract_subgraph(graph, node_ids, self.edge_kinds)
        laplacian, sub_ids = build_laplacian(subgraph, self.edge_kinds)

        if laplacian.shape[0] < self.min_partition_size:
            return Partition(
                id=partition_id,
                node_ids=node_ids,
                depth=depth,
            )

        try:
            result = compute_fiedler(laplacian, sub_ids)
        except ValueError:
            # Disconnected subgraph — shouldn't happen if we pre-split components,
            # but handle gracefully by treating as leaf
            return Partition(
                id=partition_id,
                node_ids=node_ids,
                depth=depth,
            )

        # Check connectivity threshold
        if self.connectivity_threshold > 0 and result.algebraic_connectivity > self.connectivity_threshold:
            return Partition(
                id=partition_id,
                node_ids=node_ids,
                fiedler_value=result.algebraic_connectivity,
                depth=depth,
            )

        # Split by Fiedler sign
        left_ids: set[str] = set()
        right_ids: set[str] = set()
        for nid, label in result.partition.items():
            if label == 0:
                left_ids.add(nid)
            else:
                right_ids.add(nid)

        # Guard against degenerate splits (all nodes on one side)
        if not left_ids or not right_ids:
            return Partition(
                id=partition_id,
                node_ids=node_ids,
                fiedler_value=result.algebraic_connectivity,
                depth=depth,
            )

        left_frozen = frozenset(left_ids)
        right_frozen = frozenset(right_ids)

        # Recurse
        left_child = self._partition_recursive(graph, left_frozen, f"{partition_id}.0", depth + 1)
        right_child = self._partition_recursive(graph, right_frozen, f"{partition_id}.1", depth + 1)

        return Partition(
            id=partition_id,
            node_ids=node_ids,
            children=(left_child, right_child),
            fiedler_value=result.algebraic_connectivity,
            depth=depth,
        )

    def _build_balanced_tree(
        self,
        partitions: list[Partition],
        prefix: str,
        depth: int,
    ) -> Partition:
        """Build a balanced binary tree from a list of partitions."""
        if len(partitions) == 1:
            return partitions[0]

        mid = len(partitions) // 2
        left_parts = partitions[:mid]
        right_parts = partitions[mid:]

        left_ids = frozenset().union(*(p.node_ids for p in left_parts))
        right_ids = frozenset().union(*(p.node_ids for p in right_parts))
        all_ids = left_ids | right_ids

        left_child = self._build_balanced_tree(left_parts, f"{prefix}.0", depth + 1)
        right_child = self._build_balanced_tree(right_parts, f"{prefix}.1", depth + 1)

        return Partition(
            id=prefix,
            node_ids=all_ids,
            children=(left_child, right_child),
            depth=depth,
        )

    @staticmethod
    def get_leaf_partitions(root: Partition) -> list[Partition]:
        """
        Return all leaf partitions as a flat list.

        Every node in the original graph appears in exactly one leaf.
        """
        leaves: list[Partition] = []
        stack = [root]
        while stack:
            p = stack.pop()
            if p.is_leaf:
                leaves.append(p)
            else:
                assert p.children is not None
                stack.append(p.children[0])
                stack.append(p.children[1])
        return leaves

    @staticmethod
    def find_partition(root: Partition, node_id: str) -> Partition | None:
        """
        Find which leaf partition a node belongs to.

        Returns the leaf Partition containing node_id, or None.
        """
        stack = [root]
        while stack:
            p = stack.pop()
            if node_id not in p.node_ids:
                continue
            if p.is_leaf:
                return p
            assert p.children is not None
            stack.append(p.children[0])
            stack.append(p.children[1])
        return None


# ─────────────────────────────────────────────────────────────────
# Virtual Sink/Source Augmentation
# ─────────────────────────────────────────────────────────────────


def augment_partition(
    graph: CallGraph,
    partition: Partition,
    edge_kinds: set[EdgeKind] | None = None,
) -> tuple[str, str]:
    """
    Add virtual source and sink nodes to a leaf partition.

    Virtual source connects to all entry points (nodes with no incoming
    edges from within the partition). Virtual sink is connected from all
    exit points (nodes with no outgoing edges within the partition).

    These virtual nodes enable uniform module-to-module reachability
    queries: "can module A reach module B?" becomes "can vs_A reach vt_B?"

    Args:
        graph: The full call graph (modified in place).
        partition: The leaf partition to augment.
        edge_kinds: Edge types to consider for entry/exit detection.

    Returns:
        (source_id, sink_id) — IDs of the created virtual nodes.
    """
    if edge_kinds is None:
        edge_kinds = {EdgeKind.CALLS}

    pid = partition.id
    source_id = f"vs_{pid}"
    sink_id = f"vt_{pid}"

    partition_nodes = partition.node_ids

    # Find entry points: nodes with no incoming edges from within partition
    has_internal_incoming: set[str] = set()
    has_internal_outgoing: set[str] = set()
    for edge in graph.edges:
        if edge.kind not in edge_kinds:
            continue
        if edge.source_id in partition_nodes and edge.target_id in partition_nodes:
            has_internal_incoming.add(edge.target_id)
            has_internal_outgoing.add(edge.source_id)

    entry_points = partition_nodes - has_internal_incoming
    exit_points = partition_nodes - has_internal_outgoing

    # If no entry points (all internal — cyclic), connect to all
    if not entry_points:
        entry_points = partition_nodes
    # If no exit points (all internal — cyclic), connect from all
    if not exit_points:
        exit_points = partition_nodes

    # Add virtual source node
    graph.add_node(
        GraphNode(
            id=source_id,
            kind=NodeKind.MODULE,
            name=f"source({pid})",
            metadata={"virtual": True, "role": "source", "partition": pid},
        )
    )

    # Add virtual sink node
    graph.add_node(
        GraphNode(
            id=sink_id,
            kind=NodeKind.MODULE,
            name=f"sink({pid})",
            metadata={"virtual": True, "role": "sink", "partition": pid},
        )
    )

    # Connect source → entry points
    for entry in entry_points:
        graph.add_edge(
            GraphEdge(
                source_id=source_id,
                target_id=entry,
                kind=EdgeKind.CALLS,
                confidence=1.0,
            )
        )

    # Connect exit points → sink
    for exit_node in exit_points:
        graph.add_edge(
            GraphEdge(
                source_id=exit_node,
                target_id=sink_id,
                kind=EdgeKind.CALLS,
                confidence=1.0,
            )
        )

    return source_id, sink_id
