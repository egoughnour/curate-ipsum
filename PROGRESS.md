# Curate-Ipsum — Progress Log

**Last updated:** 2026-02-08

---

## Project Goal

Curate-ipsum is a **mutation testing orchestration MCP server** that bridges LLM-generated code and formally verified patches. It uses graph-spectral decomposition, belief revision (AGM theory), and a CEGIS/CEGAR synthesis loop to transform mutation testing from a quality metric into the foundation of a verified code synthesis pipeline.

### Primary Environment

- Python 3.10+ on any POSIX-compatible system
- Dependencies: `py-brs>=2.0.0`, `pydantic>=2.0`, `mcp>=1.0.0`
- Optional: `scipy`, `networkx` (Phase 2), `z3-solver`, `sympy` (Phase 5+)

### Companion Repositories

- **py-brs** (`github.com/egoughnour/brs`, PyPI: `py-brs`): AGM-compliant belief revision library. v2.0.0 released with contraction + entrenchment. Import as `brs`.
- **curate-ipsum** (`github.com/egoughnour/curate-ipsum`): This repository.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      MCP Interface                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │  Mutation   │  │  Graph     │  │  Symbolic  │         │
│  │  Parsers    │  │  Spectral  │  │  Execution │         │
│  │  (Phase 1✓) │  │  (Phase 2) │  │  (Phase 6) │         │
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

### Phase 1: Foundation ✓ (Complete)

| Item | Status | File(s) | Notes |
|------|--------|---------|-------|
| MCP server infrastructure | ✓ | `server.py` | FastMCP-based |
| Stryker report parsing | ✓ | `parsers/stryker_parser.py` | JavaScript mutation tool |
| mutmut parser | ✓ | `parsers/mutmut_parser.py` | Python mutation tool (SQLite cache) |
| Run history + PID metrics | ✓ | `tools.py` | Precision/completeness tracking |
| Flexible region model | ✓ | `regions/models.py` | Hierarchical: file → class → func → lines |
| Framework auto-detection | ✓ | `parsers/detection.py` | Language + tool detection |
| Unified parser interface | ✓ | `parsers/__init__.py` | Routes to correct parser |

### Phase 2: Graph Infrastructure ~ (In Progress — This Is the Active Phase)

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
| Dependency graph extraction | ✗ | — | **TODO: Phase 2, Step 1** |
| Laplacian construction | ✗ | — | **TODO: Phase 2, Step 2** |
| Fiedler vector computation | ✗ | — | **TODO: Phase 2, Step 3** |
| Recursive partitioning | ✗ | — | **TODO: Phase 2, Step 4** |
| Virtual sink/source augmentation | ✗ | — | **TODO: Phase 2, Step 5** |
| Hierarchical SCC condensation | ✗ | — | **TODO: Phase 2, Step 6** |
| Planar subgraph identification | ✗ | — | **TODO: Phase 2, Step 7** |
| Kameda preprocessing (O(1) reachability) | ✗ | — | **TODO: Phase 2, Step 8** |
| MCP tools for graph queries | ✗ | — | **TODO: Phase 2, Step 9** |

### Phase 3: Multi-Framework Orchestration ~ (Partially Done)

| Item | Status | Notes |
|------|--------|-------|
| Unified mutation framework interface | ✓ | Done in Phase 1 |
| cosmic-ray parser | ✗ | Deferred — not blocking Phase 2 |
| poodle parser | ✗ | Deferred |
| universalmutator parser | ✗ | Deferred |

### Phase 4: Belief Revision Engine ~ (Partially Done)

| Item | Status | Notes |
|------|--------|-------|
| py-brs library (AGM core) | ✓ | v2.0.0 released |
| Evidence adapter (mutation→belief) | ✓ | `adapters/evidence.py` |
| Theory manager | ✓ | `theory/manager.py` |
| AGM contraction | ✓ | In py-brs v2.0.0 |
| Entrenchment calculation | ✓ | In py-brs v2.0.0 |
| Provenance DAG storage | ✗ | Depends on Phase 2 graph infra |
| Failure mode analyzer | ✗ | Depends on Provenance DAG |

### Phases 5–8: Not Started

Synthesis Loop, Verification Backends, Graph Database + RAG, Production Hardening. See `ROADMAP.md` for details.

---

## What's Next

### Phase 2: Graph-Spectral Infrastructure (Active)

See `PHASE2_PLAN.md` for the full implementation plan with 9 steps, file-by-file specs, and dependency graph.

**Blocking prerequisites:**
1. ~~Graph models~~ (done)
2. ~~Call graph extraction~~ (done)
3. ~~SCC detection~~ (done)
4. Add `scipy>=1.10` and `networkx>=3.0` to `pyproject.toml` dependencies

**Summary of work:**
1. Dependency graph extraction (`graph/dependency_extractor.py`)
2. Laplacian construction (`graph/spectral.py`)
3. Fiedler vector computation (`graph/spectral.py`)
4. Recursive partitioning (`graph/partitioner.py`)
5. Virtual sink/source augmentation (`graph/partitioner.py`)
6. Hierarchical SCC condensation (`graph/hierarchy.py`)
7. Planar subgraph identification (`graph/planarity.py`)
8. Kameda preprocessing (`graph/kameda.py`)
9. MCP tools for graph queries (`server.py`, `tools.py`)

**Integration testing:**
1. Extract call graph from curate-ipsum's own source code
2. Compute Fiedler partitioning on that graph
3. Verify SCC condensation produces valid DAG
4. Test O(1) reachability against BFS ground truth
5. Run all existing tests (`pytest tests/ -v`)

---

## Known Limitations & Open Questions

- **LPython optional**: ASR extractor requires LPython which is alpha-status. AST extractor is always available. See `docs/lpython_klee_feasibility.md`. `→ D-001`
- **Fiedler on disconnected graphs**: Need to handle disconnected components separately — Fiedler vector is undefined for disconnected graphs. `→ D-004`
- **Planarity NP-hard**: Maximal planar subgraph identification is NP-hard in general. Using networkx heuristics as mitigation. `→ D-006`
- **scipy not yet in dependencies**: Must add `scipy` and `networkx` to `pyproject.toml` before Phase 2 work begins.

---

## File Inventory

```
curate-ipsum/
├── server.py              # MCP server entry point (FastMCP)
├── tools.py               # MCP tool implementations
├── models.py              # Pydantic data models (MutationRunResult, etc.)
├── config.toml            # Server configuration
├── pyproject.toml         # Package metadata + dependencies
├── graph/
│   ├── __init__.py        # Public API (CallGraph, extractors, etc.)
│   ├── models.py          # CallGraph, GraphNode, GraphEdge, SCC, condensation
│   ├── extractor.py       # Abstract base class for extractors
│   ├── ast_extractor.py   # Python AST-based call graph extraction
│   └── asr_extractor.py   # LPython ASR-based extraction (optional)
├── parsers/
│   ├── __init__.py        # Unified parser interface
│   ├── detection.py       # Framework + language auto-detection
│   ├── stryker_parser.py  # Stryker JSON report parser
│   └── mutmut_parser.py   # mutmut SQLite cache parser
├── regions/
│   └── models.py          # Region, RegionLevel (file/class/func/lines)
├── adapters/
│   └── evidence.py        # Mutation results → BRS beliefs
├── domains/
│   └── ...                # Domain-specific logic
├── theory/
│   └── manager.py         # Theory manager for belief revision
├── tests/
│   ├── test_m1_regions.py    # Region model tests
│   ├── test_m1_parsers.py    # Parser tests (Stryker + mutmut)
│   ├── test_graph_extraction.py  # AST extractor tests
│   └── test_brs_integration.py   # BRS integration tests
├── docs/
│   ├── m1_m3_audit.md     # M1-M3 audit findings
│   └── lpython_klee_feasibility.md  # LPython/KLEE feasibility study
├── README.md              # Project overview + roadmap
├── ROADMAP.md             # Full milestone tracker (M1–M7)
├── CONTEXT.md             # Directory structure + naming conventions
├── DOCS_INDEX.md          # Documentation navigation guide
├── PROGRESS.md            # ← You are here
├── DECISIONS.md           # Architectural decision log
├── PHASE2_PLAN.md         # Phase 2 implementation plan (active)
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
