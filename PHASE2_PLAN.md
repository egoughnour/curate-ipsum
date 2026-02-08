# Phase 2: Graph-Spectral Infrastructure — Implementation Plan

**Last updated:** 2026-02-08
**Status:** ✓ Complete — all 9 steps implemented and tested (195 tests passing)
**Exit criteria:** Query reachability between any two functions in O(1) after O(n) preprocessing — **MET**

> **AMENDED 2026-02-08:** All 9 steps implemented in a single session. 6 new source files, 4 new test files, 5 MCP tools. Committed as `d34b411`.

---

## Objective

Build the graph-spectral infrastructure that transforms curate-ipsum's call graphs into a hierarchically decomposed, partitioned structure supporting O(1) reachability queries via Kameda's algorithm on planar subgraphs.

**What we're building on:** Phase 1 delivered call graph extraction (AST + ASR), a `CallGraph` class with Tarjan SCC, condensation, BFS reachability, and topological sort. See `PROGRESS.md` for the full inventory.

**What we're building:** Laplacian → Fiedler → partitioning → hierarchy → planarity → Kameda → MCP tools.

---

## Dependency Graph

```
Step 1: Dependency Graph Extraction
    │
    ▼
Step 2: Laplacian Construction ──────────────────┐
    │                                             │
    ▼                                             │
Step 3: Fiedler Vector Computation                │
    │                                             │
    ▼                                             │
Step 4: Recursive Partitioning                    │
    │                                             │
    ├──► Step 5: Virtual Sink/Source Augmentation  │
    │                                             │
    ▼                                             │
Step 6: Hierarchical SCC Condensation ◄───────────┘
    │
    ▼
Step 7: Planar Subgraph Identification
    │
    ▼
Step 8: Kameda Preprocessing
    │
    ▼
Step 9: MCP Tools for Graph Queries
```

Steps 1 and 2 can be developed in parallel. Steps 5 and 6 can overlap.

---

## Step 1: Dependency Graph Extraction

**File:** `graph/dependency_extractor.py` (new)
**Complexity:** Medium
**Dependencies:** Existing `graph/models.py`
**Est. LOC:** ~150

### What

Extract import-level dependency relationships between Python modules. This produces a module-level `CallGraph` with `IMPORTS` edges, complementing the function-level call graph from `ast_extractor.py`.

### Interface

```python
class DependencyExtractor:
    """Extract module-level dependency graph from a Python project."""

    def extract_directory(self, directory: Path) -> CallGraph:
        """Build dependency graph from all .py files in directory."""
        ...

    def extract_file(self, file_path: Path) -> List[ImportInfo]:
        """Extract imports from a single file."""
        ...
```

### Design Details

- Parse `import X` and `from X import Y` statements using Python AST.
- Resolve relative imports against the project root.
- Each module becomes a `GraphNode` with `NodeKind.MODULE`.
- Each import becomes a `GraphEdge` with `EdgeKind.IMPORTS`.
- Distinguish stdlib, third-party, and local imports (only local imports create edges).
- Edge confidence: 1.0 for direct imports, 0.7 for wildcard imports (`from X import *`).

### Acceptance Criteria

- [ ] Correctly extracts imports from curate-ipsum's own source tree
- [ ] Handles relative imports, re-exports, conditional imports
- [ ] Produces valid `CallGraph` compatible with Laplacian construction
- [ ] Unit tests covering: simple imports, relative imports, circular imports, missing modules

---

## Step 2: Laplacian Construction

**File:** `graph/spectral.py` (new)
**Complexity:** Low–Medium
**Dependencies:** `graph/models.py`, `scipy`
**Est. LOC:** ~80
**Decision:** `DECISIONS.md → D-003`

### What

Construct the graph Laplacian matrix L = D − A from a `CallGraph`, where D is the degree matrix and A is the adjacency matrix.

### Interface

```python
def build_adjacency_matrix(
    graph: CallGraph,
    edge_kinds: Optional[Set[EdgeKind]] = None,
    directed: bool = False,
) -> Tuple[scipy.sparse.csr_matrix, List[str]]:
    """
    Build sparse adjacency matrix from CallGraph.

    Returns: (matrix, node_id_list) where node_id_list[i] maps
    matrix row/col i to a node ID.
    """

def build_laplacian(
    graph: CallGraph,
    edge_kinds: Optional[Set[EdgeKind]] = None,
) -> Tuple[scipy.sparse.csr_matrix, List[str]]:
    """
    Build the symmetric graph Laplacian L = D - A.

    For directed graphs, uses the symmetrized version:
    A_sym = (A + A^T) / 2, then L = D_sym - A_sym.
    """
```

### Design Details

- Use `scipy.sparse.csr_matrix` for memory efficiency.
- Symmetrize directed edges: if A → B exists, treat as undirected for Laplacian purposes.
- Filter edges by `EdgeKind` — typically `CALLS` only, excluding `DEFINES` and `INHERITS`.
- The node-to-index mapping (`List[str]`) is returned alongside the matrix so results can be mapped back to `GraphNode` IDs.
- Weight edges by confidence: `edge.confidence` becomes the matrix entry.

### Acceptance Criteria

- [ ] L is symmetric positive semi-definite (smallest eigenvalue ≈ 0)
- [ ] Row sums of L are all zero (Laplacian property)
- [ ] Sparse format: < 5% nonzero entries for typical call graphs
- [ ] Unit tests: simple triangle graph, star graph, disconnected graph

---

## Step 3: Fiedler Vector Computation

**File:** `graph/spectral.py` (same file as Step 2)
**Complexity:** Medium
**Dependencies:** Step 2, `scipy.sparse.linalg`
**Est. LOC:** ~100
**Decision:** `DECISIONS.md → D-003`, `D-004`

### What

Compute the Fiedler vector (eigenvector of the second-smallest eigenvalue λ₂ of the Laplacian). This vector provides the optimal bipartition of the graph.

### Interface

```python
@dataclass
class FiedlerResult:
    """Result of Fiedler vector computation."""
    vector: np.ndarray          # Fiedler vector values per node
    algebraic_connectivity: float  # λ₂
    node_ids: List[str]         # Mapping: vector[i] corresponds to node_ids[i]
    partition: Dict[str, int]   # node_id → partition (0 or 1) based on sign

def compute_fiedler(
    laplacian: scipy.sparse.csr_matrix,
    node_ids: List[str],
    tolerance: float = 1e-8,
) -> FiedlerResult:
    """
    Compute the Fiedler vector of a connected graph.

    Raises:
        ValueError: If the graph is disconnected (use compute_fiedler_components instead)
    """

def compute_fiedler_components(
    graph: CallGraph,
    edge_kinds: Optional[Set[EdgeKind]] = None,
    tolerance: float = 1e-8,
) -> List[FiedlerResult]:
    """
    Compute Fiedler vectors per connected component.

    Components with < 3 nodes are returned with trivial partitions.
    """
```

### Design Details

- Use `scipy.sparse.linalg.eigsh(L, k=2, which='SM')` — iterative, efficient for sparse matrices.
- Partition by sign of Fiedler vector: `vector[i] < 0 → partition 0`, `vector[i] >= 0 → partition 1`.
- Handle numerical edge cases: if λ₂ < tolerance, the graph is effectively disconnected — decompose into components first.
- For disconnected graphs, find connected components (BFS on symmetrized adjacency), then compute Fiedler per component.
- Fallback for small graphs (< 20 nodes): use dense `numpy.linalg.eigh` for reliability.

### Acceptance Criteria

- [ ] Correct partition of a known graph (e.g., barbell graph partitions at the bridge)
- [ ] λ₂ > 0 for connected graphs, λ₂ ≈ 0 for disconnected
- [ ] Handles graphs from 3 nodes to 1000+ nodes
- [ ] Graceful fallback when `eigsh` fails to converge
- [ ] Unit tests: path graph, cycle graph, barbell graph, star graph

---

## Step 4: Recursive Partitioning

**File:** `graph/partitioner.py` (new)
**Complexity:** Medium
**Dependencies:** Steps 2–3
**Est. LOC:** ~200
**Decision:** `DECISIONS.md → D-005`

### What

Recursively apply Fiedler bipartition to produce a partition tree. Each leaf is a small, well-connected subgraph.

### Interface

```python
@dataclass
class Partition:
    """A node in the partition tree."""
    id: str
    node_ids: FrozenSet[str]
    children: Optional[Tuple["Partition", "Partition"]] = None
    fiedler_value: Optional[float] = None  # λ₂ of this subgraph
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.children is None

class GraphPartitioner:
    """Recursive Fiedler partitioner."""

    def __init__(
        self,
        min_partition_size: int = 3,
        max_depth: int = 20,
        connectivity_threshold: float = 0.01,
    ):
        ...

    def partition(self, graph: CallGraph) -> Partition:
        """Recursively bipartition the graph."""
        ...

    def get_leaf_partitions(self, root: Partition) -> List[Partition]:
        """Return all leaf partitions (flat list)."""
        ...

    def find_partition(self, root: Partition, node_id: str) -> Optional[Partition]:
        """Find which leaf partition a node belongs to."""
        ...
```

### Design Details

- Stop recursion when: `len(node_ids) < min_partition_size`, or `depth >= max_depth`, or `λ₂ > connectivity_threshold` (partition is already well-connected).
- Build subgraph `CallGraph` for each partition to compute its Laplacian.
- Partition IDs follow a binary tree scheme: root = "0", left = "0.0", right = "0.1", etc.
- The `connectivity_threshold` is tunable — a higher value produces fewer, larger partitions.

### Acceptance Criteria

- [ ] Produces balanced partitions on synthetic graphs
- [ ] Respects `min_partition_size` and `max_depth`
- [ ] Every node appears in exactly one leaf partition
- [ ] Partition tree can be traversed and queried
- [ ] Unit tests: balanced tree graph, chain graph, clique graph

---

## Step 5: Virtual Sink/Source Augmentation

**File:** `graph/partitioner.py` (same file as Step 4)
**Complexity:** Low
**Dependencies:** Step 4
**Est. LOC:** ~60
**Decision:** `DECISIONS.md → D-008`

### What

Add virtual source and sink nodes to each leaf partition for uniform reachability queries.

### Interface

```python
def augment_partition(
    graph: CallGraph,
    partition: Partition,
) -> Tuple[str, str]:
    """
    Add virtual source and sink nodes to a partition.

    Returns: (source_id, sink_id)
    """
```

### Design Details

- Virtual source `vs_{partition_id}`: edges to all entry points (in-degree 0 within partition).
- Virtual sink `vt_{partition_id}`: edges from all exit points (out-degree 0 within partition).
- Virtual nodes get `NodeKind.MODULE` kind with `metadata={"virtual": True}`.
- If a partition has no entry points (all nodes are internal), the virtual source connects to all nodes.
- If a partition has no exit points (all nodes call each other cyclically), the virtual sink connects from all nodes.

### Acceptance Criteria

- [ ] Virtual source connects to correct entry points
- [ ] Virtual sink connects from correct exit points
- [ ] Virtual nodes are distinguishable from real nodes via metadata
- [ ] Augmentation doesn't break existing graph algorithms

---

## Step 6: Hierarchical SCC Condensation

**File:** `graph/hierarchy.py` (new)
**Complexity:** Medium
**Dependencies:** Steps 4–5, existing `CallGraph.condensation()`
**Est. LOC:** ~180

### What

Build a hierarchical decomposition by alternating SCC condensation and Fiedler partitioning: condense → partition → recurse within each partition → condense again → continue until atoms.

### Interface

```python
@dataclass
class HierarchyNode:
    """Node in the hierarchical decomposition tree."""
    id: str
    level: int                     # 0 = root
    operation: str                 # "condense" or "partition"
    node_ids: FrozenSet[str]       # Original graph node IDs in this subtree
    scc_members: Optional[List[FrozenSet[str]]] = None  # If operation == "condense"
    partition_info: Optional[Partition] = None            # If operation == "partition"
    children: List["HierarchyNode"] = field(default_factory=list)

class HierarchyBuilder:
    """Build hierarchical decomposition via alternating condense/partition."""

    def __init__(
        self,
        partitioner: GraphPartitioner,
        min_scc_size: int = 2,
        max_levels: int = 10,
    ):
        ...

    def build(self, graph: CallGraph) -> HierarchyNode:
        """Build full hierarchy from a CallGraph."""
        ...

    def flatten(self, root: HierarchyNode) -> List[FrozenSet[str]]:
        """Get all leaf-level node groups."""
        ...
```

### Design Details

- Level 0: Full graph
- Level 1 (condense): Run Tarjan SCC, collapse each SCC to a single node
- Level 2 (partition): Apply Fiedler bipartition to the condensed DAG
- Level 3 (condense): Within each partition, find SCCs again
- Continue alternating until all partitions are below `min_partition_size` or are singleton SCCs.
- The hierarchy tree stores both SCC membership and partition membership at each level.

### Acceptance Criteria

- [ ] Produces valid hierarchy for graphs with and without cycles
- [ ] Condensation at each level is correct (verified against `CallGraph.strongly_connected_components()`)
- [ ] Leaf nodes cover all original nodes exactly once
- [ ] Handles graphs that are already DAGs (SCC step is a no-op)
- [ ] Unit tests: cyclic graph, DAG, mixed graph

---

## Step 7: Planar Subgraph Identification

**File:** `graph/planarity.py` (new)
**Complexity:** High
**Dependencies:** Step 6, `networkx`
**Est. LOC:** ~150
**Decision:** `DECISIONS.md → D-006`

### What

For each leaf partition, identify the maximal planar subgraph and extract Kuratowski subgraphs (K₅ or K₃,₃) as the non-planar complement.

### Interface

```python
@dataclass
class PlanarityResult:
    """Result of planarity analysis on a subgraph."""
    is_planar: bool
    planar_subgraph: CallGraph          # Maximal planar subgraph
    non_planar_edges: Set[GraphEdge]    # Edges that break planarity
    kuratowski_subgraph: Optional[CallGraph]  # K₅ or K₃,₃ if non-planar
    embedding: Optional[Dict]           # Planar embedding (networkx format)

def check_planarity(graph: CallGraph) -> PlanarityResult:
    """
    Test if a CallGraph is planar.

    Uses networkx.check_planarity() (Boyer-Myrvold, O(n)).
    If not planar, identifies a Kuratowski subgraph and computes
    a maximal planar subgraph by removing minimal edges.
    """

def callgraph_to_networkx(graph: CallGraph) -> nx.DiGraph:
    """Convert CallGraph to networkx DiGraph."""

def networkx_to_callgraph(nx_graph: nx.DiGraph, original: CallGraph) -> CallGraph:
    """Convert networkx DiGraph back to CallGraph, preserving metadata."""
```

### Design Details

- Convert `CallGraph` → networkx `DiGraph` (preserving node/edge metadata as attributes).
- Use `nx.check_planarity()` for the O(n) planarity test.
- If not planar, networkx provides the Kuratowski subgraph.
- Maximal planar subgraph: iteratively remove edges from Kuratowski subgraphs until the remainder is planar. (This is a heuristic — exact maximal planar subgraph is NP-hard.)
- The planar embedding from networkx is needed for Kameda preprocessing in Step 8.
- Non-planar edges are stored separately for fallback BFS reachability.

### Acceptance Criteria

- [ ] Correctly identifies planar graphs (trees, series-parallel graphs)
- [ ] Correctly identifies non-planar graphs (K₅, K₃,₃)
- [ ] Kuratowski extraction works
- [ ] Maximal planar subgraph is actually planar (verified by re-checking)
- [ ] Conversion between CallGraph and networkx preserves all metadata
- [ ] Unit tests: tree, K₅, K₃,₃, Petersen graph, random planar graph

---

## Step 8: Kameda Preprocessing

**File:** `graph/kameda.py` (new)
**Complexity:** High
**Dependencies:** Step 7
**Est. LOC:** ~250
**Decision:** `DECISIONS.md → D-006`

### What

Implement Kameda's 1975 algorithm for O(1) reachability queries on planar directed graphs after O(n) preprocessing.

### Interface

```python
class KamedaIndex:
    """
    O(1) reachability index for planar directed graphs.

    Based on: Kameda, T. (1975). "On the vector representation
    of the reachability in planar directed graphs."

    After O(n) preprocessing, can answer "does u reach v?" in O(1).
    """

    @classmethod
    def build(cls, graph: CallGraph, embedding: Dict) -> "KamedaIndex":
        """
        Build the reachability index.

        Args:
            graph: Must be a planar directed graph
            embedding: Planar embedding from networkx

        Raises:
            ValueError: If graph is not planar
        """
        ...

    def reaches(self, source_id: str, target_id: str) -> bool:
        """O(1) reachability query."""
        ...

    def all_reachable_from(self, source_id: str) -> Set[str]:
        """Get all nodes reachable from source (uses index, not BFS)."""
        ...
```

### Design Details

- Kameda's algorithm assigns each node a vector label such that u reaches v if and only if u's label dominates v's label (component-wise comparison).
- Preprocessing: (1) compute st-numbering of the planar graph, (2) assign vector labels via a single traversal.
- The st-numbering requires a single source and single sink — this is where virtual sink/source augmentation (Step 5) becomes essential.
- For non-planar edges (identified in Step 7), maintain a separate set and combine: `reaches(u, v) = kameda_reaches(u, v) OR any non-planar path exists`.
- Non-planar path checking falls back to BFS on the non-planar complement (typically very small).

### Acceptance Criteria

- [ ] O(1) query time (verified by timing)
- [ ] Results match BFS ground truth on 100% of test cases
- [ ] Handles graphs with virtual source/sink correctly
- [ ] Preprocessing time is O(n) (verified by timing on increasing graph sizes)
- [ ] Unit tests: DAGs, planar graphs with virtual nodes, comparison against `CallGraph.reachable_from()`

---

## Step 9: MCP Tools for Graph Queries

**File:** `server.py` (modify), `tools.py` (modify)
**Complexity:** Low–Medium
**Dependencies:** All above
**Est. LOC:** ~120

### What

Expose the graph-spectral infrastructure as MCP tools that can be called by LLM agents.

### New MCP Tools

```python
@server.tool(description="Extract and analyze the call graph of a Python project")
def extract_call_graph(workingDirectory: str, backend: str = "auto") -> dict:
    """Returns: node count, edge count, SCC count, connected components."""

@server.tool(description="Compute Fiedler partitioning of a project's call graph")
def compute_partitioning(
    workingDirectory: str,
    min_partition_size: int = 3,
    max_depth: int = 10,
) -> dict:
    """Returns: partition tree with node assignments and λ₂ values."""

@server.tool(description="Query reachability between two functions")
def query_reachability(
    workingDirectory: str,
    source_function: str,
    target_function: str,
) -> dict:
    """Returns: {reachable: bool, method: 'kameda'|'bfs', path: [...]}."""

@server.tool(description="Get the hierarchical decomposition of a project")
def get_hierarchy(workingDirectory: str) -> dict:
    """Returns: hierarchy tree with SCC and partition info at each level."""

@server.tool(description="Find which partition a function belongs to")
def find_function_partition(
    workingDirectory: str,
    function_name: str,
) -> dict:
    """Returns: partition ID, sibling functions, entry/exit points."""
```

### Acceptance Criteria

- [ ] All tools work end-to-end on curate-ipsum's own source code
- [ ] Results are JSON-serializable
- [ ] Error messages are helpful when graph extra isn't installed
- [ ] Integration test: extract → partition → query reachability pipeline

---

## New File Summary

| File | Step | Purpose | Dependencies |
|------|------|---------|--------------|
| `graph/dependency_extractor.py` | 1 | Module-level import graph | `graph/models.py` |
| `graph/spectral.py` | 2–3 | Laplacian + Fiedler computation | `scipy`, `graph/models.py` |
| `graph/partitioner.py` | 4–5 | Recursive partitioning + virtual nodes | `graph/spectral.py` |
| `graph/hierarchy.py` | 6 | Alternating condense/partition tree | `graph/partitioner.py`, `graph/models.py` |
| `graph/planarity.py` | 7 | Planar subgraph + Kuratowski extraction | `networkx`, `graph/models.py` |
| `graph/kameda.py` | 8 | O(1) reachability index | `graph/planarity.py` |
| `tests/test_spectral.py` | 2–3 | Spectral computation tests | `scipy` |
| `tests/test_partitioner.py` | 4–6 | Partitioning + hierarchy tests | — |
| `tests/test_planarity.py` | 7–8 | Planarity + Kameda tests | `networkx` |
| `tests/test_mcp_graph.py` | 9 | MCP tool integration tests | — |

---

## pyproject.toml Changes

```toml
[project.optional-dependencies]
graph = [
    "scipy>=1.10",
    "networkx>=3.0",
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "ruff>=0.1.0",
    "mypy>=1.0",
    "scipy>=1.10",      # For graph tests
    "networkx>=3.0",    # For graph tests
]
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Step |
|------|------------|--------|------------|------|
| `eigsh` convergence failure | Low | Medium | Dense fallback for small graphs; report error for large | 3 |
| Maximal planar subgraph is poor quality | Medium | Low | Accept suboptimal; Kameda still works on any planar subgraph | 7 |
| Kameda implementation bugs | Medium | High | Exhaustive comparison against BFS ground truth | 8 |
| Graph too large for in-memory processing | Low | High | Lazy evaluation; process per-component | 2–6 |
| networkx API changes | Low | Low | Pin version; thin wrapper | 7 |

---

## Estimated Timeline

| Step | Effort | Cumulative |
|------|--------|------------|
| 1. Dependency graph | 1 day | 1 day |
| 2. Laplacian | 0.5 day | 1.5 days |
| 3. Fiedler | 1 day | 2.5 days |
| 4. Partitioner | 1.5 days | 4 days |
| 5. Virtual nodes | 0.5 day | 4.5 days |
| 6. Hierarchy | 1 day | 5.5 days |
| 7. Planarity | 1.5 days | 7 days |
| 8. Kameda | 2 days | 9 days |
| 9. MCP tools | 1 day | 10 days |
| Testing + integration | 2 days | **12 days** |

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Reachability query time (Kameda) | < 1 μs | `timeit` on 1000+ queries |
| Reachability accuracy | 100% | Compare against BFS on full test suite |
| Preprocessing time | O(n) | Time vs graph size plot |
| Partition balance | < 3:1 ratio | Max partition / min partition size |
| Test coverage | > 90% | `pytest --cov` on `graph/` |

---

## Revision History

- **v1.0** (2026-02-08): Initial Phase 2 plan created from architectural vision, existing code audit, and ROADMAP analysis.
