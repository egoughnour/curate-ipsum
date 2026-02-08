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
│                  (23 tools registered)                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │  Mutation   │  │  Graph     │  │  Symbolic  │         │
│  │  Parsers    │  │  Spectral  │  │  Execution │         │
│  │  (M1 ~)    │  │  (M2 ✓)   │  │  (Phase 6) │         │
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

### M1: Multi-Framework Foundation ~ (Active — Completing Remaining Parsers)

> **AMENDED 2026-02-08:** M1 was previously listed as "Phase 1 ✓ Complete" — that reflected the core infrastructure (server, regions, Stryker, mutmut). The ROADMAP M1 exit criteria ("Run any Python mutation tool through single MCP interface") requires the remaining Python mutation tool parsers. These are now the active focus.

| Item | Status | File(s) | Notes |
|------|--------|---------|-------|
| MCP server infrastructure | ✓ | `server.py` | FastMCP-based, 23 tools |
| Stryker report parsing | ✓ | `parsers/stryker_parser.py` | JavaScript mutation tool |
| mutmut parser | ✓ | `parsers/mutmut_parser.py` | Python mutation tool (SQLite cache) |
| Run history + PID metrics | ✓ | `tools.py` | Precision/completeness tracking |
| Flexible region model | ✓ | `regions/models.py` | Hierarchical: file → class → func → lines |
| Framework auto-detection | ✓ | `parsers/detection.py` | Language + tool detection |
| Unified parser interface | ✓ | `parsers/__init__.py` | Routes to correct parser |
| cosmic-ray parser | ✗ | — | **TODO: Active** |
| poodle parser | ✗ | — | **TODO: Active** |
| universalmutator parser | ✗ | — | **TODO: Active** |

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

### M3: Belief Revision Engine ~ (Partially Done)

| Item | Status | Notes |
|------|--------|-------|
| py-brs library (AGM core) | ✓ | v2.0.0 released |
| Evidence adapter (mutation→belief) | ✓ | `adapters/evidence_adapter.py` |
| Theory manager | ✓ | `theory/manager.py` |
| AGM contraction | ✓ | In py-brs v2.0.0 |
| Entrenchment calculation | ✓ | In py-brs v2.0.0 |
| Provenance DAG storage | ✗ | Can now build on M2 graph infra |
| Failure mode analyzer | ✗ | Depends on Provenance DAG |

### Phases 4–8: Not Started

Synthesis Loop, Verification Backends, Graph Database + RAG, Production Hardening. See `ROADMAP.md` for details.

---

## What's Next

### M1 Completion: Remaining Python Mutation Parsers (Active)

**Exit criteria:** Run any Python mutation tool through single MCP interface.

**Remaining work (3 parsers):**

1. **cosmic-ray parser** (`parsers/cosmic_ray_parser.py`) — JSON report format. Detection signals already exist in `detection.py` (`.cosmic-ray.toml`, confidence 0.8).
2. **poodle parser** (`parsers/poodle_parser.py`) — JSON report format. Detection signals needed.
3. **universalmutator parser** (`parsers/universalmutator_parser.py`) — Multi-language tool, typically text/JSON output. Detection signals needed.

**Integration points to update:**

- `parsers/__init__.py` — route new tool names to new parsers (currently raises `UnsupportedFrameworkError`)
- `parsers/detection.py` — add detection signals for poodle and universalmutator
- Tests in `tests/` — one test file per parser

**No server.py changes needed** — `run_mutation_tests_tool` already uses the unified `parse_mutation_output()` interface.

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
| `tests/test_graph_extraction.py` | 26 | AST extractor, call resolution |
| `tests/test_brs_integration.py` | 15 (skipped) | BRS evidence adapter integration |
| `tests/test_spectral.py` | 41 | Laplacian, Fiedler, partitioner, virtual nodes |
| `tests/test_planarity_kameda.py` | 54 | Planarity, Kameda reachability, BFS verification |
| `tests/test_hierarchy_deps.py` | 48 | Hierarchy, dependency extractor, imports |
| `tests/test_mcp_graph.py` | 26 | MCP graph tools end-to-end pipeline |
| **Total** | **265 passed, 15 skipped** | |

---

## File Inventory

```
curate-ipsum/
├── server.py              # MCP server entry point (23 tools)
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
│   ├── __init__.py        # Unified parser interface
│   ├── detection.py       # Framework + language auto-detection
│   ├── stryker_parser.py  # Stryker JSON report parser
│   └── mutmut_parser.py   # mutmut SQLite cache parser
├── regions/
│   └── models.py          # Region, RegionLevel (file/class/func/lines)
├── adapters/
│   └── evidence_adapter.py  # Mutation results → BRS beliefs
├── theory/
│   └── manager.py         # Theory manager for belief revision
├── tests/
│   ├── test_m1_regions.py       # Region model tests
│   ├── test_m1_parsers.py       # Parser tests (Stryker + mutmut)
│   ├── test_graph_extraction.py # AST extractor tests
│   ├── test_brs_integration.py  # BRS integration tests
│   ├── test_spectral.py         # Laplacian/Fiedler/partitioner tests
│   ├── test_planarity_kameda.py # Planarity + Kameda tests
│   ├── test_hierarchy_deps.py   # Hierarchy + dependency tests
│   └── test_mcp_graph.py        # MCP graph tool integration tests
├── docs/
│   ├── m1_m3_audit.md     # M1-M3 audit findings
│   └── lpython_klee_feasibility.md  # LPython/KLEE feasibility study
├── README.md              # Project overview + roadmap
├── ROADMAP.md             # Full milestone tracker (M1–M7)
├── CONTEXT.md             # Directory structure + naming conventions
├── DOCS_INDEX.md          # Documentation navigation guide
├── PROGRESS.md            # ← You are here
├── DECISIONS.md           # Architectural decision log (D-001 through D-009)
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
