# Curate-Ipsum — Progress Log

**Last updated:** 2026-02-08

---

## Project Goal

Curate-ipsum is a **mutation testing orchestration MCP server** that bridges LLM-generated code and formally verified patches. It uses graph-spectral decomposition, belief revision (AGM theory), and a CEGIS/CEGAR synthesis loop to transform mutation testing from a quality metric into the foundation of a verified code synthesis pipeline.

### Primary Environment

- Python 3.10+ on any POSIX-compatible system
- Dependencies: `py-brs>=2.0.0`, `pydantic>=2.0`, `mcp>=1.0.0`
- Optional: `scipy>=1.10`, `networkx>=3.0` (graph extras), `z3-solver`, `sympy` (Phase 5+)

### Companion Repositories

- **py-brs** (`github.com/egoughnour/brs`, PyPI: `py-brs`): AGM-compliant belief revision library. v2.0.0 released with contraction + entrenchment. Import as `brs`.
- **curate-ipsum** (`github.com/egoughnour/curate-ipsum`): This repository.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      MCP Interface                       │
│                  (35 tools registered)                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │  Mutation   │  │  Graph     │  │  Symbolic  │         │
│  │  Parsers    │  │  Spectral  │  │  Execution │         │
│  │  (M1 ✓)    │  │  (M2 ✓)   │  │  (Phase 6) │         │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘         │
│         │               │               │                │
│         ▼               ▼               ▼                │
│  ┌───────────────────────────────────────────────┐       │
│  │         Belief Revision Engine (Phase 4)      │       │
│  │         py-brs: AGM theory, entrenchment      │       │
│  └───────────────────┬───────────────────────────┘       │
│                      │                                   │
│                      ▼                                   │
│  ┌───────────────────────────────────────────────┐       │
│  │         Synthesis Loop (Phase 5)              │       │
│  │         CEGIS + CEGAR + Genetic Algorithm     │       │
│  └───────────────────────────────────────────────┘       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Full vision: `architectural_vision.md`. Decisions: `DECISIONS.md`.

---

## Current Status

### M1: Multi-Framework Foundation ✓ (Complete)

> **AMENDED 2026-02-08:** All 5 mutation framework parsers implemented. M1 exit criteria met: "Run any Python mutation tool through single MCP interface."

| Item | Status | File(s) | Notes |
|------|--------|---------|-------|
| MCP server infrastructure | ✓ | `server.py` | FastMCP-based, 35 tools |
| Stryker report parsing | ✓ | `parsers/stryker_parser.py` | JavaScript mutation tool |
| mutmut parser | ✓ | `parsers/mutmut_parser.py` | Python mutation tool (SQLite cache) |
| Run history + PID metrics | ✓ | `tools.py` | Precision/completeness tracking |
| Flexible region model | ✓ | `regions/models.py` | Hierarchical: file → class → func → lines |
| Framework auto-detection | ✓ | `parsers/detection.py` | Language + tool detection |
| Unified parser interface | ✓ | `parsers/__init__.py` | Routes to correct parser |
| cosmic-ray parser | ✓ | `parsers/cosmic_ray_parser.py` | JSON dump + SQLite session DB |
| poodle parser | ✓ | `parsers/poodle_parser.py` | JSON mutation-testing-report-schema |
| universalmutator parser | ✓ | `parsers/universalmutator_parser.py` | Plain text killed.txt / not-killed.txt |

### M2: Graph-Spectral Infrastructure ✓ (Complete)

> **AMENDED 2026-02-08:** All 9 steps from `PHASE2_PLAN.md` implemented. 195 tests passing. Committed as `d34b411`.

| Item | Status | File(s) | Notes |
|------|--------|---------|-------|
| Graph models (CallGraph, Node, Edge) | ✓ | `graph/models.py` | Backend-agnostic, serializable |
| Call graph extraction (AST) | ✓ | `graph/ast_extractor.py` | Two-pass: definitions → calls |
| ASR extractor (LPython) | ✓ | `graph/asr_extractor.py` | Optional, requires LPython |
| Tarjan SCC detection | ✓ | `graph/models.py` | `strongly_connected_components()` |
| Graph condensation (DAG of SCCs) | ✓ | `graph/models.py` | `condensation()` method |
| BFS reachability | ✓ | `graph/models.py` | `reachable_from()`, `reaches()` |
| Topological sort | ✓ | `graph/models.py` | For DAG ordering |
| DOT export | ✓ | `graph/models.py` | Graphviz visualization |
| Dependency graph extraction | ✓ | `graph/dependency_extractor.py` | Module-level import graphs |
| Laplacian construction | ✓ | `graph/spectral.py` | Sparse L = D − A, symmetrized |
| Fiedler vector computation | ✓ | `graph/spectral.py` | `eigsh` + dense fallback `→ D-003` |
| Recursive partitioning | ✓ | `graph/partitioner.py` | Binary tree of Fiedler bipartitions |
| Virtual sink/source augmentation | ✓ | `graph/partitioner.py` | `augment_partition()` `→ D-008` |
| Hierarchical SCC condensation | ✓ | `graph/hierarchy.py` | Alternating condense/partition `→ D-005` |
| Planar subgraph identification | ✓ | `graph/planarity.py` | Boyer-Myrvold + Kuratowski `→ D-006` |
| Kameda O(1) reachability | ✓ | `graph/kameda.py` | 2D dominance labels + BFS fallback |
| MCP tools for graph queries | ✓ | `server.py` | 5 tools: extract, partition, reach, hierarchy, find |

### M3: Belief Revision Engine ✓ (Complete)

> **AMENDED 2026-02-08:** All M3 items implemented. Exit criteria met: "Track belief evolution across synthesis attempts with full provenance." 32 MCP tools total (8 new M3 tools)

| Item | Status | File(s) | Notes |
|------|--------|---------|-------|
| py-brs library (AGM core) | ✓ | PyPI `py-brs` | v2.0.0 released |
| Evidence adapter (mutation→belief) | ✓ | `adapters/evidence_adapter.py` | Mutation results → BRS Evidence |
| Theory manager | ✓ | `theory/manager.py` | High-level API, provenance wired |
| AGM contraction | ✓ | In py-brs v2.0.0 | 3 strategies: entrenchment, minimal, full_cascade |
| Entrenchment calculation | ✓ | In py-brs v2.0.0 | `compute_entrenchment()` |
| Typed assertion model | ✓ | `theory/assertions.py` | 6 kinds + contradiction detection → D-010 |
| Provenance DAG | ✓ | `theory/provenance.py` | Append-only causal chain → D-010 |
| Rollback mechanism | ✓ | `theory/rollback.py` | Checkpoints + undo via provenance DAG |
| Failure mode analyzer | ✓ | `theory/failure_analyzer.py` | 7 failure modes, heuristic classification → D-011 |
| MCP tools (8 new) | ✓ | `server.py` | store_evidence, provenance, why_believe, stability, rollback, undo, analyze_failure, world_history |

### M4: Synthesis Loop ✓ (Complete)

> **AMENDED 2026-02-08:** All M4 items implemented. Exit criteria met: "Generate patch that kills previously-surviving mutant, verified correct."

| Item | Status | File(s) | Notes |
|------|--------|---------|-------|
| Synthesis data models | ✓ | `synthesis/models.py` | Individual, CodePatch, Specification, Counterexample, SynthesisResult |
| Abstract LLM client | ✓ | `synthesis/llm_client.py` | ABC + MockLLMClient + prompt builder |
| Cloud LLM backend | ✓ | `synthesis/cloud_llm.py` | Anthropic + OpenAI via httpx → D-012 |
| Local LLM backend | ✓ | `synthesis/local_llm.py` | Ollama HTTP API → D-012 |
| Population management | ✓ | `synthesis/population.py` | Elite/tournament selection, add/remove |
| Fitness evaluation | ✓ | `synthesis/fitness.py` | CE avoidance + spec satisfaction - complexity → D-013 |
| AST-aware crossover | ✓ | `synthesis/ast_operators.py` | Subtree swap, directed mutation |
| Entropy manager | ✓ | `synthesis/entropy.py` | Shannon entropy, diversity injection |
| CEGIS engine | ✓ | `synthesis/cegis.py` | Full loop: LLM → GA → verify → CE feedback |
| MCP tools (4 new) | ✓ | `server.py` | synthesize_patch, synthesis_status, cancel_synthesis, list_synthesis_runs |

### M6: Graph Persistence (Partial — Graph Storage Complete, RAG Deferred)

> **AMENDED 2026-02-08:** Graph persistence layer implemented. 56 new tests, 35 MCP tools total. Storage package with SQLite (primary) + Kuzu (optional) backends, incremental update engine, and synthesis result persistence. RAG/embeddings deferred to follow-up.

| Item | Status | File(s) | Notes |
|------|--------|---------|-------|
| Abstract GraphStore ABC | ✓ | `storage/graph_store.py` | Factory pattern mirrors D-012 → D-014 |
| SQLite graph store (primary) | ✓ | `storage/sqlite_graph_store.py` | 7 tables, WAL mode, zero deps |
| Kuzu graph store (optional) | ✓ | `storage/kuzu_graph_store.py` | Cypher queries, embedded graph DB |
| Synthesis result persistence | ✓ | `storage/synthesis_store.py` | JSONL append-only, multi-project |
| Kameda index persistence | ✓ | `storage/sqlite_graph_store.py` | O(1) reachability survives restart |
| Fiedler partition persistence | ✓ | `storage/sqlite_graph_store.py` | Materialized path encoding |
| Incremental update engine | ✓ | `storage/incremental.py` | SHA-256 file hashing → D-015 |
| MCP tools (3 new) | ✓ | `server.py` | incremental_update, persistent_graph_stats, graph_query |
| Server wiring | ✓ | `server.py` | extract_call_graph, compute_partitioning, synthesize_patch persist automatically |
| Code embedding / RAG | ⚪ | - | Deferred to follow-up |

### Phases 5, 7–8: Not Started

Verification Backends, Production Hardening. See `ROADMAP.md` for details.

---

## What's Next

### RAG + Embeddings (M6 Follow-Up)

**Exit criteria:** Natural language queries over codebase with graph-backed retrieval.

**Key tasks:** Code embedding model, vector index, semantic search, text-to-Cypher pipeline.

### M5: Verification Backends

**Exit criteria:** Verify patch correctness against specification with proof certificate.

**Key tasks:** Z3 integration, CEGAR abstraction levels, SymPy path conditions, KLEE container. See `ROADMAP.md` for details.

---

## Known Limitations & Open Questions

- **LPython optional**: ASR extractor requires LPython which is alpha-status. AST extractor is always available. See `docs/lpython_klee_feasibility.md`. `→ D-001`
- ~~**Fiedler on disconnected graphs**: Need to handle disconnected components separately.~~ **Resolved** in `graph/spectral.py` via per-component Fiedler. `→ D-004`
- ~~**Planarity NP-hard**: Maximal planar subgraph identification is NP-hard in general.~~ **Mitigated** via iterative edge removal heuristic in `graph/planarity.py`. `→ D-006`
- ~~**scipy not yet in dependencies**~~ **Resolved**: Added as `[graph]` optional dependency. `→ D-007`

---

## Test Suite Summary

| Test File | Count | Covers |
|-----------|-------|--------|
| `tests/test_m1_regions.py` | 25 | Region model parsing, containment, overlap |
| `tests/test_m1_parsers.py` | 25 | Stryker + mutmut parsing, detection |
| `tests/test_new_parsers.py` | 62 | cosmic-ray, poodle, universalmutator parsers |
| `tests/test_graph_extraction.py` | 26 | AST extractor, call resolution |
| `tests/test_brs_integration.py` | 15 | BRS evidence adapter integration |
| `tests/test_spectral.py` | 41 | Laplacian, Fiedler, partitioner, virtual nodes |
| `tests/test_planarity_kameda.py` | 54 | Planarity, Kameda reachability, BFS verification |
| `tests/test_hierarchy_deps.py` | 48 | Hierarchy, dependency extractor, imports |
| `tests/test_mcp_graph.py` | 26 | MCP graph tools end-to-end pipeline |
| `tests/test_assertions.py` | 38 | Typed assertions, serialization, contradiction detection |
| `tests/test_provenance.py` | 30 | Provenance DAG, event recording, path queries |
| `tests/test_rollback.py` | 12 | Rollback, checkpoints, undo operations |
| `tests/test_failure_analyzer.py` | 38 | Failure classification, overfitting, contraction suggestions |
| `tests/test_m3_end_to_end.py` | 9 | M3 full lifecycle: evidence → assertion → provenance → rollback |
| `tests/test_synthesis_models.py` | 28 | Synthesis data models, config validation |
| `tests/test_llm_client.py` | 18 | Mock/cloud/local LLM clients, prompt building |
| `tests/test_genetic_operators.py` | 27 | Population, fitness, AST operators, entropy |
| `tests/test_cegis.py` | 8 | CEGIS engine, cancellation, timeout |
| `tests/test_m4_end_to_end.py` | 11 | M4 full pipeline end-to-end |
| `tests/test_synthesis_store.py` | 10 | Synthesis store JSONL persistence |
| `tests/test_sqlite_graph_store.py` | 25 | SQLite graph store round-trips, queries |
| `tests/test_kuzu_graph_store.py` | 12 | Kuzu graph store (skipped if kuzu not installed) |
| `tests/test_incremental.py` | 15 | Incremental update engine, change detection |
| `tests/test_m6_end_to_end.py` | 7 | M6 full pipeline: persist → query → update |
| **Total** | **616 passed, 1 pre-existing failure, 1 skipped** | |

---

## File Inventory

```
curate-ipsum/
├── server.py              # MCP server entry point (35 tools)
├── tools.py               # Async test/mutation execution layer
├── models.py              # Pydantic data models (MutationRunResult, etc.)
├── config.toml            # Server configuration
├── pyproject.toml         # Package metadata + dependencies
├── graph/
│   ├── __init__.py        # Public API + optional dependency flags
│   ├── models.py          # CallGraph, GraphNode, GraphEdge, SCC, condensation
│   ├── extractor.py       # Abstract base class for extractors
│   ├── ast_extractor.py   # Python AST-based call graph extraction
│   ├── asr_extractor.py   # LPython ASR-based extraction (optional)
│   ├── dependency_extractor.py  # Module-level import graph extraction
│   ├── spectral.py        # Laplacian + Fiedler vector computation
│   ├── partitioner.py     # Recursive Fiedler partitioning + virtual nodes
│   ├── hierarchy.py       # Alternating condense/partition tree
│   ├── planarity.py       # Boyer-Myrvold planarity + Kuratowski
│   └── kameda.py          # O(1) reachability index (2D dominance)
├── parsers/
│   ├── __init__.py        # Unified parser interface (routes 5 frameworks)
│   ├── detection.py       # Framework + language auto-detection
│   ├── stryker_parser.py  # Stryker JSON report parser
│   ├── mutmut_parser.py   # mutmut SQLite cache parser
│   ├── cosmic_ray_parser.py   # cosmic-ray JSON dump + SQLite parser
│   ├── poodle_parser.py       # poodle JSON mutation-testing-report parser
│   └── universalmutator_parser.py  # universalmutator text file parser
├── regions/
│   └── models.py          # Region, RegionLevel (file/class/func/lines)
├── adapters/
│   └── evidence_adapter.py  # Mutation results → BRS beliefs
├── theory/
│   ├── __init__.py            # Package with submodule listing
│   ├── manager.py             # Theory manager (provenance + rollback wired)
│   ├── assertions.py          # Typed assertion model + contradiction detection
│   ├── provenance.py          # Append-only provenance DAG
│   ├── rollback.py            # Rollback manager + checkpoints
│   └── failure_analyzer.py    # Heuristic failure classification
├── synthesis/
│   ├── __init__.py            # Public API + optional dependency flag
│   ├── models.py              # Individual, CodePatch, Specification, SynthesisResult
│   ├── llm_client.py          # LLMClient ABC + MockLLMClient + prompt builder
│   ├── cloud_llm.py           # Cloud LLM (Anthropic/OpenAI) via httpx
│   ├── local_llm.py           # Local LLM (Ollama) via httpx
│   ├── population.py          # GA population management
│   ├── fitness.py             # Fitness evaluation (CE + spec - complexity)
│   ├── ast_operators.py       # AST crossover + directed mutation
│   ├── entropy.py             # Shannon entropy + diversity injection
│   └── cegis.py               # CEGIS engine (main synthesis loop)
├── storage/
│   ├── __init__.py            # Package init, exports
│   ├── synthesis_store.py     # JSONL persistence for synthesis results
│   ├── graph_store.py         # Abstract GraphStore ABC + factory → D-014
│   ├── sqlite_graph_store.py  # SQLite backend (primary, zero deps)
│   ├── kuzu_graph_store.py    # Kuzu backend (optional, Cypher queries)
│   └── incremental.py         # File hash tracking + delta updates → D-015
├── tests/
│   ├── test_m1_regions.py       # Region model tests
│   ├── test_m1_parsers.py       # Parser tests (Stryker + mutmut)
│   ├── test_new_parsers.py      # cosmic-ray, poodle, universalmutator tests
│   ├── test_graph_extraction.py # AST extractor tests
│   ├── test_brs_integration.py  # BRS integration tests
│   ├── test_assertions.py       # Typed assertions + contradiction detection
│   ├── test_provenance.py       # Provenance DAG tests
│   ├── test_rollback.py         # Rollback + checkpoint tests
│   ├── test_failure_analyzer.py # Failure classification tests
│   ├── test_m3_end_to_end.py    # M3 full lifecycle E2E
│   ├── test_spectral.py         # Laplacian/Fiedler/partitioner tests
│   ├── test_planarity_kameda.py # Planarity + Kameda tests
│   ├── test_hierarchy_deps.py   # Hierarchy + dependency tests
│   ├── test_mcp_graph.py        # MCP graph tool integration tests
│   ├── test_synthesis_models.py   # Synthesis data model tests
│   ├── test_llm_client.py        # LLM client tests
│   ├── test_genetic_operators.py  # GA operator tests
│   ├── test_cegis.py             # CEGIS engine tests
│   ├── test_m4_end_to_end.py     # M4 full pipeline E2E
│   ├── test_synthesis_store.py    # Synthesis JSONL store tests
│   ├── test_sqlite_graph_store.py # SQLite graph store tests
│   ├── test_kuzu_graph_store.py   # Kuzu graph store tests (skip if no kuzu)
│   ├── test_incremental.py        # Incremental update engine tests
│   └── test_m6_end_to_end.py     # M6 full pipeline E2E
├── docs/
│   ├── m1_m3_audit.md     # M1-M3 audit findings
│   └── lpython_klee_feasibility.md  # LPython/KLEE feasibility study
├── README.md              # Project overview + roadmap
├── ROADMAP.md             # Full milestone tracker (M1–M7)
├── CONTEXT.md             # Directory structure + naming conventions
├── DOCS_INDEX.md          # Documentation navigation guide
├── PROGRESS.md            # ← You are here
├── DECISIONS.md           # Architectural decision log (D-001 through D-015)
├── PHASE2_PLAN.md         # Phase 2 implementation plan (complete)
├── architectural_vision.md       # Graph-spectral framework theory
├── synthesis_framework.md        # CEGIS/CEGAR/genetic approach
├── belief_revision_framework.md  # AGM theory + provenance
├── m1_multi_framework_plan.md    # Phase 1 implementation plan (done)
├── brs_integration_plan.md       # BRS ↔ curate-ipsum mapping
├── brs_v2_refactoring_plan.md    # py-brs v2.0.0 plan
├── brs_contract_pr.md            # AGM contraction spec
├── brs_cicd.md                   # CI/CD pipeline docs
├── summary.md                    # Functionality catalog
├── potential_directions.md       # Enhancement ideas
├── synergies.md                  # Tool ecosystem integration
└── inferred_goals.md             # Evidence-based goal hierarchy
```

---

## Revision History

- **v1.0** (2026-02-08): Initial PROGRESS.md created from comprehensive codebase audit. Phase 1 complete, Phase 2 active.
- **v2.0** (2026-02-08): Phase 2 (M2) complete — all 9 steps implemented (195 tests). M1 remaining parsers now active focus. Updated architecture diagram, file inventory, test summary, and known limitations.
- **v3.0** (2026-02-08): M1 ✅ complete (3 new parsers: cosmic-ray, poodle, universalmutator). M3 ✅ complete (assertions, provenance DAG, rollback, failure analyzer, 8 new MCP tools). 468 tests passing. Updated all inventories.
- **v4.0** (2026-02-08): M4 ✅ complete (synthesis loop: CEGIS + genetic algorithm + LLM client). 560 tests passing. 10 new synthesis files, 5 new test files, 4 new MCP tools.
- **v5.0** (2026-02-08): M6 🟡 partial (graph persistence: SQLite + Kuzu backends, incremental updates, synthesis persistence). 616 tests passing. 6 new storage files, 5 new test files, 3 new MCP tools (35 total). RAG deferred.
