# Decision Log

Every significant architectural and implementation decision, with reasoning and context. Written for cold-start — assumes the reader has zero prior context.

**Last updated:** 2026-02-08

---

## D-001: Use Dual AST/ASR Extractors with AST as Default

**Date:** 2026-01-27
**Status:** Active
**Affects:** `graph/ast_extractor.py`, `graph/asr_extractor.py`, `graph/extractor.py`

**Context:** Curate-ipsum needs to extract call graphs from Python source code for graph-spectral analysis (Fiedler partitioning, reachability queries). LPython's ASR (Abstract Semantic Representation) provides richer semantic data including resolved types and imports, but LPython is alpha-status and requires type-annotated code. Python's built-in `ast` module is always available but provides only syntactic information.

**Decision:** Implement both extractors behind an abstract `CallGraphExtractor` base class. AST backend is always available and is the default. ASR backend is optional and auto-selected when LPython is detected. Both produce the same `CallGraph` data structure.

**Reasoning:**
- AST is universally available — no external dependency needed for basic functionality.
- ASR provides higher-confidence call resolution (resolved imports, type information), valuable for Fiedler partitioning accuracy.
- The abstract interface (`extractor.py`) means downstream code (spectral analysis, partitioning) is backend-agnostic.
- LPython's alpha status means we can't depend on it for production use. See `docs/lpython_klee_feasibility.md`.

**Upgrade path:** When LPython stabilizes, ASR can become the default. The LLVM backend may eventually enable the LPython → C → KLEE symbolic execution pipeline.

---

## D-002: CallGraph as Central Data Structure for All Graph Operations

**Date:** 2026-01-27
**Status:** Active
**Affects:** `graph/models.py`, all Phase 2 modules

**Context:** Multiple Phase 2 operations (Laplacian construction, Fiedler computation, SCC detection, reachability) all operate on graph data. The question is whether to use networkx graphs directly, define custom models, or both.

**Decision:** Define a custom `CallGraph` class in `graph/models.py` with typed `GraphNode` and `GraphEdge` objects. Built-in graph algorithms (SCC, reachability, condensation, topological sort) operate directly on `CallGraph`. Phase 2 spectral methods will convert to scipy sparse matrices as needed.

**Reasoning:**
- Custom models carry domain-specific metadata (source locations, signatures, confidence scores, edge kinds) that networkx's generic node/edge attributes would lose type safety on.
- `CallGraph` already has index structures (`_outgoing`, `_incoming`, `_by_file`) optimized for common queries.
- Serialization (`to_dict()`, `from_dict()`, `to_dot()`) is cleaner with custom models.
- Conversion to scipy sparse matrices for eigenvalue computation is a one-line operation (`scipy.sparse.csr_matrix(adjacency)`).

**Trade-off:** Algorithms like planarity testing will require conversion to networkx format, since we shouldn't reimplement networkx's planarity algorithms.

---

## D-003: Laplacian and Fiedler Computation via scipy.sparse

**Date:** 2026-02-08
**Status:** Active
**Affects:** `graph/spectral.py` (new file)

**Context:** The Fiedler vector (second eigenvector of the graph Laplacian L = D − A) is the core of Phase 2's spectral partitioning. For production codebases, call graphs can have thousands of nodes. The Laplacian is typically sparse (most functions call only a few others), so dense eigenvalue decomposition would waste memory and time.

**Decision:** Use `scipy.sparse.linalg.eigsh` on a `scipy.sparse.csr_matrix` Laplacian. Restrict computation to the two smallest eigenvalues (`k=2, which='SM'`). Handle disconnected components by computing Fiedler vectors per connected component.

**Reasoning:**
- `eigsh` uses ARPACK (iterative Arnoldi method) — O(n·k) for k eigenvectors, efficient for sparse matrices.
- Call graphs are naturally sparse: a function with 5 callees in a 1000-node graph produces a matrix that is 99.5% zeros.
- Connected component handling is essential: the Fiedler vector is undefined for disconnected graphs (λ₂ = 0 with multiplicity > 1). Each connected component gets its own Fiedler partitioning.

**Fallback:** If `eigsh` fails to converge (rare, but possible for near-singular Laplacians), fall back to dense `numpy.linalg.eigh` for small graphs (< 500 nodes) or report the failure for large ones.

---

## D-004: Per-Component Fiedler Partitioning for Disconnected Graphs

**Date:** 2026-02-08
**Status:** Active
**Affects:** `graph/spectral.py`, `graph/partitioner.py` (new files)

**Context:** Real-world codebases often have disconnected call graphs — utility modules that are imported but don't call each other, test files that are independent. The algebraic connectivity λ₂ is 0 for disconnected graphs, making the Fiedler vector meaningless.

**Decision:** Before Fiedler computation, decompose the graph into connected components. Apply Fiedler partitioning independently to each component with ≥ 3 nodes. Components with 1–2 nodes are treated as singleton partitions.

**Reasoning:**
- This is the mathematically correct approach — Fiedler's theorem assumes connectivity.
- Most connected components in a codebase will be non-trivial (tens to hundreds of nodes), so partitioning is still valuable.
- Singleton components (isolated utility functions) don't benefit from spectral analysis.

---

## D-005: Hierarchical Decomposition via Alternating Condense/Partition

**Date:** 2026-02-08
**Status:** Active
**Affects:** `graph/hierarchy.py` (new file), `graph/partitioner.py`

**Context:** The architectural vision calls for a hierarchical decomposition strategy: condense SCCs into single nodes, then apply Fiedler partitioning to the condensed DAG, then recurse within each partition. This produces a tree of progressively finer partitions, terminating at Kuratowski atoms (K₅, K₃,₃ subgraphs) or singleton nodes.

**Decision:** Implement a `HierarchyBuilder` that alternates: (1) SCC condensation (using existing `CallGraph.condensation()`), (2) Fiedler bipartition on the resulting DAG, (3) recurse into each partition. Stop recursion when a partition is below `min_partition_size` (default: 3) or when the Fiedler value λ₂ exceeds a threshold (indicating the partition is already well-connected).

**Reasoning:**
- Alternating condense/partition respects both cyclic structure (via SCC) and spectral structure (via Fiedler).
- The condensation step guarantees the partitioner always operates on a DAG, avoiding issues with directed cycles in Fiedler computation.
- Recursion depth is bounded by `log₂(n)` in practice (each bipartition halves the graph).
- This matches the architectural vision in `architectural_vision.md` § A.6.

**Trade-off:** The alternating strategy may over-partition tightly coupled code. The `min_partition_size` parameter provides a tuning knob.

---

## D-006: Use networkx for Planarity Testing, Custom Code for Kameda

**Date:** 2026-02-08
**Status:** Active
**Affects:** `graph/planarity.py`, `graph/kameda.py` (new files)

**Context:** O(1) reachability queries require Kameda's algorithm, which only works on planar directed graphs. We need to: (1) identify the maximal planar subgraph, and (2) implement Kameda preprocessing on it. Planarity testing and maximal planar subgraph extraction are well-solved problems available in networkx. Kameda's algorithm is specialized and not available in any Python library.

**Decision:** Convert `CallGraph` to networkx `DiGraph` for planarity testing using `networkx.check_planarity()`. Extract Kuratowski subgraphs (K₅, K₃,₃) as the non-planar complement. Implement Kameda's O(n) preprocessing and O(1) query as custom code in `graph/kameda.py`, following the original 1975 paper.

**Reasoning:**
- networkx has a well-tested, O(n) Boyer-Myrvold planarity implementation — no reason to reimplement.
- Kameda's algorithm is obscure enough that no Python library implements it. The original paper provides pseudocode that is straightforward to implement.
- Converting between `CallGraph` and networkx `DiGraph` is low-cost (iterate nodes and edges, O(n + m)).
- Kuratowski subgraph extraction (networkx provides this as part of planarity checking) identifies the atomic non-planar units — useful for downstream analysis.

**Trade-off:** networkx becomes a required dependency for Phase 2 (currently optional).

**Fallback:** If planarity-based O(1) reachability proves unnecessary for practical graph sizes, BFS/DFS reachability (already implemented in `CallGraph.reachable_from()`) is O(n + m) per query and may suffice.

---

## D-007: Add scipy and networkx as Phase 2 Dependencies

**Date:** 2026-02-08
**Status:** Active
**Affects:** `pyproject.toml`

**Context:** Phase 2 requires sparse eigenvalue computation (scipy) and planarity testing (networkx). Currently neither is in the project's dependency list.

**Decision:** Add `scipy>=1.10` and `networkx>=3.0` as optional dependencies under a `[project.optional-dependencies.graph]` extra, so users who only need Phase 1 (mutation parsing) don't pull in heavy numerical libraries.

**Reasoning:**
- scipy is ~30 MB; networkx is ~3 MB. Users who only need mutation score reporting shouldn't be forced to install them.
- Optional dependency pattern matches how py-brs handles optional features.
- Code should fail gracefully with a clear error message if the graph extra isn't installed.

**Upgrade path:** If Phase 2 becomes core functionality, move these to required dependencies.

---

## D-008: Virtual Sink/Source Augmentation per Module

**Date:** 2026-02-08
**Status:** Active
**Affects:** `graph/partitioner.py`

**Context:** The architectural vision calls for virtual source nodes (edges to all entry points) and virtual sink nodes (edges from all exit points) for each module/partition. This enables uniform reachability queries and module-to-module flow analysis.

**Decision:** After partitioning, augment each partition with a virtual source `s_{partition_id}` connected to nodes with no incoming edges from within the partition (entry points), and a virtual sink `t_{partition_id}` connected from nodes with no outgoing edges within the partition (exit points). Virtual nodes have a special `NodeKind` or metadata flag to distinguish them from real code nodes.

**Reasoning:**
- Uniform interface: any reachability query can be phrased as "can s_A reach t_B?" for module-level flow.
- Entry/exit detection is straightforward: entry = in-degree 0 within partition, exit = out-degree 0 within partition.
- Virtual nodes don't affect spectral properties (they're added after partitioning).

---

## Decision Index

| ID | Short Name | Date | Status |
|----|-----------|------|--------|
| D-001 | Dual AST/ASR extractors | 2026-01-27 | Active |
| D-002 | CallGraph as central data structure | 2026-01-27 | Active |
| D-003 | Sparse Laplacian + Fiedler via scipy | 2026-02-08 | Active |
| D-004 | Per-component Fiedler for disconnected graphs | 2026-02-08 | Active |
| D-005 | Alternating condense/partition hierarchy | 2026-02-08 | Active |
| D-006 | networkx for planarity, custom Kameda | 2026-02-08 | Active |
| D-007 | scipy/networkx as optional dependencies | 2026-02-08 | Active |
| D-008 | Virtual sink/source augmentation | 2026-02-08 | Active |

---

## Revision History

- **v1.0** (2026-02-08): Initial decision log created. D-001 and D-002 documented retroactively from existing code. D-003 through D-008 are Phase 2 design decisions.
