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

## D-009: Standardized Parser Interface for All Mutation Frameworks

**Date:** 2026-02-08
**Status:** Active
**Affects:** `parsers/__init__.py`, all `parsers/*_parser.py` modules

**Context:** M1 exit criteria require running "any Python mutation tool through single MCP interface." The project already has Stryker (JS) and mutmut (Python) parsers. Three more Python mutation frameworks — cosmic-ray, poodle, and universalmutator — need parsers. The MCP server's `run_mutation_tests_tool` already routes through the unified `parse_mutation_output()` interface, so new parsers only need to implement the standard entry point and be wired into the router.

**Decision:** Every parser module exports a single entry-point function with this signature:

```python
def parse_<tool>_output(
    working_directory: str,
    report_path: Optional[str] = None,
) -> Tuple[int, int, int, int, float, List[FileMutationStats]]:
```

Returning `(total, killed, survived, no_coverage, score, by_file)`. The `parsers/__init__.py` router normalizes tool name variations (e.g., `"cosmic-ray"` → `"cosmic_ray"`) and imports lazily. Each parser also exports a `find_<tool>_report()` locator function. Detection signals go in `parsers/detection.py`.

**Reasoning:**
- Uniform 6-tuple return means no downstream changes needed — `tools.py`, `server.py`, and `models.py` are already wired.
- Lazy imports avoid ImportError if a parser's optional dependencies aren't installed.
- Detection and parsing are separate concerns: detection tells you which tool ran; parsing reads the output.

---

## D-010: Provenance DAG as Append-Only Event Log

**Date:** 2026-02-08
**Status:** Active
**Affects:** `theory/provenance.py`, `theory/manager.py`, `theory/rollback.py`

**Context:** M3 requires tracking WHY each world state exists — not just WHAT it looks like (already handled by CASStore's content-addressed snapshots). We need a causal chain: which evidence triggered which revision, which assertions were added or removed, and why.

**Decision:** Implement a `ProvenanceDAG` as an append-only event log stored alongside CASStore objects. Events record `(event_type, assertion_id, evidence_id, from_world_hash, to_world_hash, strategy, reason, nodes_removed, nodes_added)`. The DAG structure is derived from event ordering (no explicit parent pointers). Serialized to CASStore as a single object per domain.

**Reasoning:**
- Append-only log provides an audit trail — events are immutable once recorded.
- Content-addressed storage means all historical worlds are already preserved; provenance just adds the "why."
- Storing the entire DAG as one object simplifies persistence (vs. storing individual events).
- The rollback mechanism only changes the world_label pointer — it doesn't copy or mutate any worlds, which is correct because CAS preserves everything.
- Indexes by assertion_id and to_world_hash enable efficient queries (why_believe, when_added, when_removed, belief_stability).

**Trade-off:** Storing the entire DAG as one object means write amplification (the whole DAG is re-serialized on each event). For typical synthesis sessions with <1000 events, this is negligible. If scaling becomes an issue, events could be stored individually with an index object.

---

## D-011: Heuristic Failure Classification (Formal Verification Deferred to M5)

**Date:** 2026-02-08
**Status:** Active
**Affects:** `theory/failure_analyzer.py`, `theory/manager.py`

**Context:** When a synthesis attempt fails, the system needs to understand WHY it failed to guide the next iteration. Formal verification (SMT/CEGAR) is planned for M5, but M3 needs failure classification now to connect failure modes to belief revision operations (which assertions to contract).

**Decision:** Implement heuristic failure classification using regex pattern matching on error messages, combined with statistical detection of overfitting (high test pass + low mutation kill) and underfitting (low test pass). Map failure modes to assertion kinds: TYPE_MISMATCH → contract TYPE assertions, POSTCONDITION_VIOLATION → contract POSTCONDITION assertions, UNDERFITTING → contract all assertions in region. Sort contraction candidates by confidence (weakest first — contract least-entrenched beliefs).

**Reasoning:**
- Pattern matching on error messages works surprisingly well for common failure modes (TypeError, AssertionError, IndexError, etc.).
- Overfitting/underfitting detection from test pass rate vs mutation kill rate is a well-established technique in mutation testing literature.
- The mapping from failure mode to assertion kind provides actionable guidance: "this failed because of a type issue, so contract the type-related beliefs."
- Heuristic approach is fast (no SMT solver), deterministic, and easy to debug.
- Formal verification in M5 will replace/augment these heuristics with proof-based classification.

**Trade-off:** Heuristic classification has lower precision than formal verification. The `confidence` field on `FailureAnalysis` communicates this uncertainty to downstream consumers. The SEMANTIC_DRIFT detection (test pass rate 50–80% with no specific error pattern) is particularly low confidence (0.5).

---

## D-012: Hybrid LLM Client with Abstract Interface

**Date:** 2026-02-08
**Status:** Active
**Affects:** `synthesis/llm_client.py`, `synthesis/cloud_llm.py`, `synthesis/local_llm.py`

**Context:** M4 requires LLM-generated code candidates as seeds for the genetic algorithm. Two options: cloud APIs (Claude/GPT) offer higher quality (85-92% syntactically valid) but cost $0.80-2/run and require API keys. Local models (Ollama + codellama:7b) are free but produce lower quality (60-70% valid) and require GPU for usable speed.

**Decision:** Implement both behind an abstract `LLMClient` ABC, mirroring D-001's dual extractor pattern. Three concrete implementations: `CloudLLMClient` (Anthropic/OpenAI via httpx), `LocalLLMClient` (Ollama HTTP API), and `MockLLMClient` (testing). Backend is selectable at runtime via `SynthesisConfig.llm_backend`.

**Reasoning:**
- Mirrors the project's established pattern (D-001: dual AST/ASR extractors behind abstract base).
- ~50 extra lines of abstraction enables runtime switching and zero-dependency testing.
- Cloud API for high-quality initial seeds; local model for cost-free refinement iterations.
- CI/CD runs with MockLLMClient — no API keys or GPU needed for testing.
- Users without GPU or API keys can still use whichever backend is available.

**Trade-off:** httpx becomes an optional dependency for cloud/local backends. Mock backend works without it.

---

## D-013: Fitness Function Formula for Genetic Algorithm

**Date:** 2026-02-08
**Status:** Active
**Affects:** `synthesis/fitness.py`, `synthesis/cegis.py`

**Context:** The genetic algorithm needs a fitness function to evaluate candidate patches. The function must balance three concerns: (1) avoiding known counterexamples, (2) satisfying the specification (tests pass), and (3) code simplicity.

**Decision:** Fitness = (0.4 × CE_avoidance) + (0.5 × spec_satisfaction) - (0.1 × complexity_penalty). CE_avoidance is the fraction of counterexamples NOT triggered. Spec_satisfaction is the fraction of test commands that pass. Complexity_penalty is AST node count / 100, capped at 1.0.

**Reasoning:**
- Spec satisfaction dominates (0.5 weight) because passing tests is the primary goal.
- CE avoidance has second priority (0.4) because avoiding known failures is critical for convergence.
- Complexity penalty is low (0.1) — a tiebreaker to prefer simpler code, not a primary concern.
- AST node count is a simple, fast proxy for complexity (no external dependency needed).
- The formula matches `synthesis_framework.md` §Genetic Algorithm Population Management.

**Trade-off:** Static weights may not be optimal for all codebases. Could be made adaptive in future milestones.

---

## D-014: Abstract GraphStore with SQLite/Kuzu Dual Backends

**Date:** 2026-02-08
**Status:** Active
**Affects:** `storage/graph_store.py`, `storage/sqlite_graph_store.py`, `storage/kuzu_graph_store.py`

**Context:** M6 requires persistent graph storage for call graphs, Kameda reachability indices, and Fiedler partition trees. Prior to M6, all graph data was in-memory and lost on restart. Synthesis results from M4 were similarly ephemeral.

**Decision:** Abstract `GraphStore` ABC with two concrete backends: SQLite (primary, zero external deps) and Kuzu (optional embedded graph DB with Cypher). Mirrors the D-012 pattern (hybrid client with abstract base + concrete backends). Factory function `build_graph_store(backend, project_path)` selects backend at runtime.

**Reasoning:**
- SQLite is stdlib — zero-dependency primary backend ensures curate-ipsum works out of the box.
- Kuzu provides native Cypher query language and efficient multi-hop traversals — important for future RAG and text-to-Cypher features.
- Storing both backends behind an ABC means new backends (Neo4j, JanusGraph) can be added without changing consumer code.
- Storage location: `.curate_ipsum/` directory alongside existing `beliefs.db`.

**Alternatives Considered:**
- Neo4j server: Rejected — external server dependency too heavy for embedded tool.
- Joern CPG + Neo4j: Deferred — Joern adds JVM dependency; Kuzu provides similar Cypher without JVM.
- SQLite only: Would work but blocks future Cypher-based queries.

---

## D-015: Incremental Graph Updates via File Hashing

**Date:** 2026-02-08
**Status:** Active
**Affects:** `storage/incremental.py`, `storage/sqlite_graph_store.py`

**Context:** Full call graph re-extraction is expensive for large projects. Users modify a few files between queries, making full re-extraction wasteful.

**Decision:** SHA-256 file hashing with change detection. The `IncrementalEngine` computes hashes for all `.py` files, compares with stored hashes, and produces a `ChangeSet` (added/modified/removed files). Only affected nodes and edges are updated. File hashes are persisted in the graph store's `file_hashes` table.

**Reasoning:**
- SHA-256 is fast enough for file-level change detection and avoids false positives.
- File-level granularity balances precision vs. complexity — function-level diffing would require AST parsing before change detection.
- Removed files trigger node/edge deletion. Modified files trigger delete + re-extract. Added files trigger extraction only.
- Full rebuild is always available as a fallback via `force_full_rebuild()`.

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
| D-009 | Standardized parser interface | 2026-02-08 | Active |
| D-010 | Provenance DAG as append-only event log | 2026-02-08 | Active |
| D-011 | Heuristic failure classification | 2026-02-08 | Active |
| D-012 | Hybrid LLM client with abstract interface | 2026-02-08 | Active |
| D-013 | Fitness function formula for GA | 2026-02-08 | Active |
| D-014 | Abstract GraphStore with SQLite/Kuzu backends | 2026-02-08 | Active |
| D-015 | Incremental graph updates via file hashing | 2026-02-08 | Active |

---

## Revision History

- **v1.0** (2026-02-08): Initial decision log created. D-001 and D-002 documented retroactively from existing code. D-003 through D-008 are Phase 2 design decisions.
- **v1.1** (2026-02-08): Added D-009 (standardized parser interface) for M1 completion. All Phase 2 decisions (D-003 through D-008) confirmed as implemented and tested.
- **v1.2** (2026-02-08): Added D-010 (provenance DAG) and D-011 (heuristic failure classification) for M3 completion.
- **v1.3** (2026-02-08): Added D-012 (hybrid LLM client) and D-013 (fitness function formula) for M4 completion.
- **v1.4** (2026-02-08): Added D-014 (abstract GraphStore) and D-015 (incremental updates) for M6 graph persistence.
