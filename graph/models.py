"""
Graph models for call graph representation.

These models are backend-agnostic and can be populated from either
Python's AST or LPython's ASR.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum


class NodeKind(str, Enum):
    """Kind of node in the call graph."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    LAMBDA = "lambda"
    COMPREHENSION = "comprehension"


class EdgeKind(str, Enum):
    """Kind of edge in the call graph."""

    CALLS = "calls"  # Direct function call
    DEFINES = "defines"  # Class/module defines function
    INHERITS = "inherits"  # Class inheritance
    IMPORTS = "imports"  # Module import
    REFERENCES = "references"  # Variable reference (weaker than call)


@dataclass(frozen=True)
class SourceLocation:
    """Source code location."""

    file: str
    line_start: int
    line_end: int
    col_start: int = 0
    col_end: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line_start}"

    def contains_line(self, line: int) -> bool:
        """Check if a line number falls within this location."""
        return self.line_start <= line <= self.line_end


@dataclass(frozen=True)
class FunctionSignature:
    """Function signature information."""

    name: str
    params: tuple[str, ...] = ()
    return_type: str | None = None
    decorators: tuple[str, ...] = ()
    is_async: bool = False
    is_generator: bool = False

    @property
    def arity(self) -> int:
        """Number of parameters."""
        return len(self.params)


@dataclass
class GraphNode:
    """Node in the call graph."""

    id: str  # Fully qualified name: module.class.function
    kind: NodeKind
    name: str  # Short name
    location: SourceLocation | None = None
    signature: FunctionSignature | None = None
    docstring: str | None = None
    metadata: dict[str, any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return False
        return self.id == other.id


@dataclass(frozen=True)
class GraphEdge:
    """Edge in the call graph."""

    source_id: str
    target_id: str
    kind: EdgeKind
    location: SourceLocation | None = None  # Where the call/reference occurs
    is_conditional: bool = False  # Inside if/try/loop
    is_dynamic: bool = False  # Dynamic call (getattr, etc.)
    confidence: float = 1.0  # 1.0 = certain, <1.0 = inferred

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.kind))


@dataclass
class CallGraph:
    """
    Directed graph representing function calls and definitions.

    This is the core data structure for M2 graph-spectral analysis.
    Designed to be populated from either Python AST or LPython ASR.
    """

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: set[GraphEdge] = field(default_factory=set)

    # Index structures for efficient queries
    _outgoing: dict[str, set[str]] = field(default_factory=dict)
    _incoming: dict[str, set[str]] = field(default_factory=dict)
    _by_file: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

        if node.id not in self._outgoing:
            self._outgoing[node.id] = set()
        if node.id not in self._incoming:
            self._incoming[node.id] = set()

        if node.location:
            file_key = node.location.file
            if file_key not in self._by_file:
                self._by_file[file_key] = set()
            self._by_file[file_key].add(node.id)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self.edges.add(edge)

        if edge.source_id not in self._outgoing:
            self._outgoing[edge.source_id] = set()
        self._outgoing[edge.source_id].add(edge.target_id)

        if edge.target_id not in self._incoming:
            self._incoming[edge.target_id] = set()
        self._incoming[edge.target_id].add(edge.source_id)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_callees(self, node_id: str) -> set[str]:
        """Get IDs of functions called by node_id."""
        return self._outgoing.get(node_id, set())

    def get_callers(self, node_id: str) -> set[str]:
        """Get IDs of functions that call node_id."""
        return self._incoming.get(node_id, set())

    def get_edges_from(self, node_id: str, kind: EdgeKind | None = None) -> Iterator[GraphEdge]:
        """Get all edges originating from a node."""
        for edge in self.edges:
            if edge.source_id == node_id:
                if kind is None or edge.kind == kind:
                    yield edge

    def get_edges_to(self, node_id: str, kind: EdgeKind | None = None) -> Iterator[GraphEdge]:
        """Get all edges pointing to a node."""
        for edge in self.edges:
            if edge.target_id == node_id:
                if kind is None or edge.kind == kind:
                    yield edge

    def get_nodes_in_file(self, file_path: str) -> set[str]:
        """Get all node IDs in a specific file."""
        return self._by_file.get(file_path, set())

    def functions(self) -> Iterator[GraphNode]:
        """Iterate over all function/method nodes."""
        for node in self.nodes.values():
            if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.LAMBDA):
                yield node

    def classes(self) -> Iterator[GraphNode]:
        """Iterate over all class nodes."""
        for node in self.nodes.values():
            if node.kind == NodeKind.CLASS:
                yield node

    def modules(self) -> Iterator[GraphNode]:
        """Iterate over all module nodes."""
        for node in self.nodes.values():
            if node.kind == NodeKind.MODULE:
                yield node

    # ─────────────────────────────────────────────────────────────────
    # Graph algorithms (M2 foundation)
    # ─────────────────────────────────────────────────────────────────

    def reachable_from(self, node_id: str, max_depth: int | None = None) -> set[str]:
        """
        Get all nodes reachable from node_id via calls.

        Args:
            node_id: Starting node
            max_depth: Maximum traversal depth (None = unlimited)

        Returns:
            Set of reachable node IDs (excluding start node)
        """
        visited: set[str] = set()
        frontier = [(node_id, 0)]

        while frontier:
            current, depth = frontier.pop()
            if current in visited:
                continue
            if current != node_id:
                visited.add(current)
            if max_depth is not None and depth >= max_depth:
                continue
            for callee in self.get_callees(current):
                if callee not in visited:
                    frontier.append((callee, depth + 1))

        return visited

    def reaches(self, node_id: str, max_depth: int | None = None) -> set[str]:
        """
        Get all nodes that can reach node_id via calls.

        Args:
            node_id: Target node
            max_depth: Maximum traversal depth (None = unlimited)

        Returns:
            Set of node IDs that can reach target (excluding target)
        """
        visited: set[str] = set()
        frontier = [(node_id, 0)]

        while frontier:
            current, depth = frontier.pop()
            if current in visited:
                continue
            if current != node_id:
                visited.add(current)
            if max_depth is not None and depth >= max_depth:
                continue
            for caller in self.get_callers(current):
                if caller not in visited:
                    frontier.append((caller, depth + 1))

        return visited

    def strongly_connected_components(self) -> list[frozenset[str]]:
        """
        Find strongly connected components using Tarjan's algorithm.

        Returns:
            List of SCCs, each as a frozenset of node IDs
        """
        index_counter = [0]
        stack: list[str] = []
        lowlinks: dict[str, int] = {}
        index: dict[str, int] = {}
        on_stack: set[str] = set()
        sccs: list[frozenset[str]] = []

        def strongconnect(node: str) -> None:
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)

            for callee in self.get_callees(node):
                if callee not in index:
                    strongconnect(callee)
                    lowlinks[node] = min(lowlinks[node], lowlinks[callee])
                elif callee in on_stack:
                    lowlinks[node] = min(lowlinks[node], index[callee])

            if lowlinks[node] == index[node]:
                scc: set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.add(w)
                    if w == node:
                        break
                sccs.append(frozenset(scc))

        for node in self.nodes:
            if node not in index:
                strongconnect(node)

        return sccs

    def topological_sort(self) -> list[str]:
        """
        Topological sort of nodes (only valid for DAG).

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If graph contains cycles
        """
        in_degree: dict[str, int] = dict.fromkeys(self.nodes, 0)
        for edge in self.edges:
            if edge.kind == EdgeKind.CALLS:
                if edge.target_id in in_degree:
                    in_degree[edge.target_id] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for callee in self.get_callees(node):
                if callee in in_degree:
                    in_degree[callee] -= 1
                    if in_degree[callee] == 0:
                        queue.append(callee)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles - not a DAG")

        return result

    def condensation(self) -> "CallGraph":
        """
        Create condensation graph (DAG of SCCs).

        Each SCC becomes a single node in the condensation graph.
        Useful for hierarchical decomposition (M2).

        Returns:
            New CallGraph where each node is an SCC
        """
        sccs = self.strongly_connected_components()
        node_to_scc: dict[str, int] = {}

        for i, scc in enumerate(sccs):
            for node in scc:
                node_to_scc[node] = i

        condensed = CallGraph()

        # Create SCC nodes
        for i, scc in enumerate(sccs):
            scc_id = f"scc_{i}"
            members = sorted(scc)
            condensed.add_node(
                GraphNode(
                    id=scc_id,
                    kind=NodeKind.MODULE,  # SCC as pseudo-module
                    name=f"SCC({', '.join(members[:3])}{'...' if len(members) > 3 else ''})",
                    metadata={"members": list(scc), "size": len(scc)},
                )
            )

        # Create edges between SCCs
        seen_edges: set[tuple[int, int]] = set()
        for edge in self.edges:
            if edge.kind == EdgeKind.CALLS:
                src_scc = node_to_scc.get(edge.source_id)
                tgt_scc = node_to_scc.get(edge.target_id)
                if src_scc is not None and tgt_scc is not None and src_scc != tgt_scc:
                    if (src_scc, tgt_scc) not in seen_edges:
                        seen_edges.add((src_scc, tgt_scc))
                        condensed.add_edge(
                            GraphEdge(
                                source_id=f"scc_{src_scc}",
                                target_id=f"scc_{tgt_scc}",
                                kind=EdgeKind.CALLS,
                            )
                        )

        return condensed

    # ─────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind.value,
                    "name": n.name,
                    "location": {
                        "file": n.location.file,
                        "line_start": n.location.line_start,
                        "line_end": n.location.line_end,
                    }
                    if n.location
                    else None,
                    "signature": {
                        "name": n.signature.name,
                        "params": list(n.signature.params),
                        "return_type": n.signature.return_type,
                        "is_async": n.signature.is_async,
                    }
                    if n.signature
                    else None,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "kind": e.kind.value,
                    "confidence": e.confidence,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CallGraph":
        """Deserialize from dictionary."""
        graph = cls()

        for n in data.get("nodes", []):
            location = None
            if n.get("location"):
                loc = n["location"]
                location = SourceLocation(
                    file=loc["file"],
                    line_start=loc["line_start"],
                    line_end=loc["line_end"],
                )

            signature = None
            if n.get("signature"):
                sig = n["signature"]
                signature = FunctionSignature(
                    name=sig["name"],
                    params=tuple(sig.get("params", [])),
                    return_type=sig.get("return_type"),
                    is_async=sig.get("is_async", False),
                )

            graph.add_node(
                GraphNode(
                    id=n["id"],
                    kind=NodeKind(n["kind"]),
                    name=n["name"],
                    location=location,
                    signature=signature,
                )
            )

        for e in data.get("edges", []):
            graph.add_edge(
                GraphEdge(
                    source_id=e["source"],
                    target_id=e["target"],
                    kind=EdgeKind(e["kind"]),
                    confidence=e.get("confidence", 1.0),
                )
            )

        return graph

    def to_dot(self, title: str = "Call Graph") -> str:
        """Export to Graphviz DOT format."""
        lines = [
            f'digraph "{title}" {{',
            "  rankdir=TB;",
            "  node [shape=box, style=filled, fillcolor=lightblue];",
        ]

        # Nodes
        for node in self.nodes.values():
            label = node.name
            if node.signature:
                params = ", ".join(node.signature.params[:3])
                if len(node.signature.params) > 3:
                    params += "..."
                label = f"{node.name}({params})"

            color = {
                NodeKind.MODULE: "lightgray",
                NodeKind.CLASS: "lightyellow",
                NodeKind.FUNCTION: "lightblue",
                NodeKind.METHOD: "lightgreen",
                NodeKind.LAMBDA: "pink",
            }.get(node.kind, "white")

            lines.append(f'  "{node.id}" [label="{label}", fillcolor={color}];')

        # Edges
        for edge in self.edges:
            style = "solid" if edge.confidence >= 0.9 else "dashed"
            color = {
                EdgeKind.CALLS: "black",
                EdgeKind.DEFINES: "blue",
                EdgeKind.INHERITS: "red",
                EdgeKind.IMPORTS: "gray",
            }.get(edge.kind, "black")
            lines.append(f'  "{edge.source_id}" -> "{edge.target_id}" [style={style}, color={color}];')

        lines.append("}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"CallGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
