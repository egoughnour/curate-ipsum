"""
Hierarchical decomposition via alternating SCC condensation and Fiedler partitioning.

Produces a tree where each level alternates between:
1. CONDENSE: Collapse strongly connected components into single nodes
2. PARTITION: Apply Fiedler bipartition to the resulting DAG

This continues recursively until partitions are below a minimum size
or consist of singleton SCCs.

References:
    architectural_vision.md § A.6
    DECISIONS.md → D-005
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

from .models import CallGraph, EdgeKind, GraphEdge, GraphNode, NodeKind
from .partitioner import GraphPartitioner, Partition
from .spectral import _extract_subgraph


@dataclass
class HierarchyNode:
    """
    Node in the hierarchical decomposition tree.

    Each node represents either a condensation step (collapsing SCCs)
    or a partitioning step (Fiedler bipartition).
    """

    id: str
    """Unique identifier for this hierarchy node."""

    level: int
    """Depth in the hierarchy tree. 0 = root."""

    operation: str
    """'condense' or 'partition' — which operation produced this level."""

    node_ids: FrozenSet[str]
    """Original graph node IDs in this subtree (all descendants included)."""

    scc_members: Optional[List[FrozenSet[str]]] = None
    """If operation == 'condense': list of SCCs found at this level."""

    partition_info: Optional[Partition] = None
    """If operation == 'partition': the Partition tree from Fiedler splitting."""

    children: List["HierarchyNode"] = field(default_factory=list)
    """Child hierarchy nodes."""

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class HierarchyBuilder:
    """
    Build hierarchical decomposition via alternating condense/partition.

    Algorithm:
        Level 0 (condense): Run Tarjan SCC on full graph → DAG of SCCs
        Level 1 (partition): Fiedler bipartition on condensed DAG
        Level 2 (condense): Within each partition, find SCCs again
        Level 3 (partition): Fiedler on each sub-DAG
        ... continue until atoms
    """

    def __init__(
        self,
        partitioner: Optional[GraphPartitioner] = None,
        min_scc_size: int = 2,
        max_levels: int = 10,
        edge_kinds: Optional[Set[EdgeKind]] = None,
    ):
        """
        Args:
            partitioner: GraphPartitioner instance (created with defaults if None).
            min_scc_size: Skip condensation for SCCs smaller than this.
            max_levels: Maximum hierarchy depth.
            edge_kinds: Edge types to use for both SCC and partitioning.
        """
        self.edge_kinds = edge_kinds or {EdgeKind.CALLS}
        self.partitioner = partitioner or GraphPartitioner(edge_kinds=self.edge_kinds)
        self.min_scc_size = min_scc_size
        self.max_levels = max_levels

    def build(self, graph: CallGraph) -> HierarchyNode:
        """
        Build the full hierarchical decomposition.

        Args:
            graph: The call graph to decompose.

        Returns:
            Root HierarchyNode of the decomposition tree.
        """
        all_ids = frozenset(graph.nodes.keys())
        if not all_ids:
            return HierarchyNode(
                id="h_0",
                level=0,
                operation="condense",
                node_ids=frozenset(),
            )

        return self._decompose(graph, all_ids, level=0, prefix="h_0")

    def _decompose(
        self,
        graph: CallGraph,
        node_ids: FrozenSet[str],
        level: int,
        prefix: str,
    ) -> HierarchyNode:
        """Recursive alternating condense/partition."""
        if level >= self.max_levels or len(node_ids) < self.min_scc_size:
            return HierarchyNode(
                id=prefix,
                level=level,
                operation="leaf",
                node_ids=node_ids,
            )

        if level % 2 == 0:
            # Even levels: CONDENSE (SCC detection)
            return self._condense_step(graph, node_ids, level, prefix)
        else:
            # Odd levels: PARTITION (Fiedler bipartition)
            return self._partition_step(graph, node_ids, level, prefix)

    def _condense_step(
        self,
        graph: CallGraph,
        node_ids: FrozenSet[str],
        level: int,
        prefix: str,
    ) -> HierarchyNode:
        """
        Condensation step: find SCCs and collapse them.

        Produces a DAG where each SCC becomes a single node.
        If all SCCs are singletons (graph is already a DAG), skip
        directly to partitioning.
        """
        subgraph = _extract_subgraph(graph, node_ids, self.edge_kinds)
        sccs = subgraph.strongly_connected_components()

        # Filter to only include SCCs within our node set
        sccs = [scc for scc in sccs if scc.issubset(node_ids)]

        # Check if condensation is useful (any non-singleton SCC?)
        non_trivial_sccs = [scc for scc in sccs if len(scc) >= self.min_scc_size]

        node = HierarchyNode(
            id=prefix,
            level=level,
            operation="condense",
            node_ids=node_ids,
            scc_members=sccs,
        )

        if not non_trivial_sccs:
            # Already a DAG — skip to partition at next level
            child = self._decompose(graph, node_ids, level + 1, f"{prefix}_p")
            node.children.append(child)
            return node

        # Build condensed graph
        condensed = subgraph.condensation()

        # Create a mapping from SCC index to original node sets
        scc_map: Dict[str, FrozenSet[str]] = {}
        for scc_node in condensed.nodes.values():
            members = scc_node.metadata.get("members", [])
            scc_map[scc_node.id] = frozenset(members)

        # Partition the condensed DAG at the next level
        partition_result = self._decompose(
            condensed,
            frozenset(condensed.nodes.keys()),
            level + 1,
            f"{prefix}_p",
        )

        # Map partition results back to original nodes
        # Each leaf in the partition tree references SCC node IDs;
        # we need to expand those back to original node IDs
        expanded = self._expand_partition_to_original(
            partition_result, scc_map, graph, level + 2, prefix
        )

        node.children = expanded
        return node

    def _partition_step(
        self,
        graph: CallGraph,
        node_ids: FrozenSet[str],
        level: int,
        prefix: str,
    ) -> HierarchyNode:
        """
        Partitioning step: apply Fiedler bipartition.

        Splits the node set into two groups based on spectral analysis,
        then recurses into each group at the next (condense) level.
        """
        subgraph = _extract_subgraph(graph, node_ids, self.edge_kinds)
        partition_tree = self.partitioner.partition(subgraph)

        node = HierarchyNode(
            id=prefix,
            level=level,
            operation="partition",
            node_ids=node_ids,
            partition_info=partition_tree,
        )

        # Get leaf partitions and recurse into each at the next level
        leaves = GraphPartitioner.get_leaf_partitions(partition_tree)

        for i, leaf in enumerate(leaves):
            if len(leaf.node_ids) <= 1:
                # Singleton — terminal
                child = HierarchyNode(
                    id=f"{prefix}_{i}",
                    level=level + 1,
                    operation="leaf",
                    node_ids=leaf.node_ids,
                )
            else:
                child = self._decompose(
                    graph, leaf.node_ids, level + 1, f"{prefix}_{i}"
                )
            node.children.append(child)

        return node

    def _expand_partition_to_original(
        self,
        partition_node: HierarchyNode,
        scc_map: Dict[str, FrozenSet[str]],
        original_graph: CallGraph,
        base_level: int,
        prefix: str,
    ) -> List[HierarchyNode]:
        """
        Expand a hierarchy over condensed nodes back to original graph nodes.

        Walks the partition hierarchy and replaces SCC node references
        with their original member sets, then recurses for further decomposition.
        """
        leaves = self._collect_leaves(partition_node)
        children: List[HierarchyNode] = []

        for i, leaf in enumerate(leaves):
            # Expand SCC IDs to original node IDs
            original_ids: Set[str] = set()
            for scc_id in leaf.node_ids:
                if scc_id in scc_map:
                    original_ids.update(scc_map[scc_id])
                else:
                    original_ids.add(scc_id)

            frozen = frozenset(original_ids)

            if len(frozen) <= 1:
                children.append(HierarchyNode(
                    id=f"{prefix}_e{i}",
                    level=base_level,
                    operation="leaf",
                    node_ids=frozen,
                ))
            else:
                # Continue decomposition on the expanded node set
                child = self._decompose(
                    original_graph, frozen, base_level, f"{prefix}_e{i}"
                )
                children.append(child)

        return children

    def _collect_leaves(self, node: HierarchyNode) -> List[HierarchyNode]:
        """Collect all leaf nodes from a hierarchy subtree."""
        if node.is_leaf:
            return [node]
        result: List[HierarchyNode] = []
        for child in node.children:
            result.extend(self._collect_leaves(child))
        return result

    def flatten(self, root: HierarchyNode) -> List[FrozenSet[str]]:
        """
        Get all leaf-level node groups from the hierarchy.

        Returns a list of frozensets — each is a group of original
        node IDs at the lowest level of decomposition.
        """
        leaves = self._collect_leaves(root)
        return [leaf.node_ids for leaf in leaves if leaf.node_ids]

    def summary(self, root: HierarchyNode) -> Dict:
        """
        Produce a summary of the hierarchy for serialization/display.

        Returns a nested dict with operation types, sizes, and children.
        """
        result: Dict = {
            "id": root.id,
            "level": root.level,
            "operation": root.operation,
            "size": len(root.node_ids),
        }

        if root.scc_members is not None:
            result["scc_count"] = len(root.scc_members)
            result["non_trivial_sccs"] = sum(
                1 for scc in root.scc_members if len(scc) >= 2
            )

        if root.partition_info is not None:
            result["fiedler_value"] = root.partition_info.fiedler_value

        if root.children:
            result["children"] = [self.summary(c) for c in root.children]

        return result
