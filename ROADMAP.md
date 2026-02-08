# Curate-Ipsum Roadmap

## Vision

Transform mutation testing from a quality metric into the foundation of a **verified code synthesis pipeline** where LLM outputs become seeds for formally proven patches.

## Current Status

**Last Updated**: 2026-02-08

| Milestone | Status | Progress |
|-----------|--------|----------|
| M1: Multi-Framework Foundation | ✅ Complete | 100% (5 parsers, 389 tests) |
| M2: Graph-Spectral Infrastructure | ✅ Complete | 100% (195 tests passing) |
| M3: Belief Revision Engine | ✅ Complete | 100% (127 new tests, 37 MCP tools) |
| M4: Synthesis Loop | ⚪ Not Started | 0% |
| M5: Verification Backends | ⚪ Not Started | 0% |
| M6: Graph Database + RAG | ⚪ Not Started | 0% |
| M7: Production Hardening | ⚪ Not Started | 0% |

## Milestones

### M1: Multi-Framework Foundation (Q1)
**Goal**: Unified interface across mutation testing tools

| Task | Status | Complexity | Dependencies |
|------|--------|------------|--------------|
| Flexible region model | ✅ Done | Medium | - |
| Stryker parser extraction | ✅ Done | Low | - |
| mutmut parser | ✅ Done | Low | - |
| Framework auto-detection | ✅ Done | Low | - |
| Unified parser interface | ✅ Done | Low | All parsers |
| cosmic-ray parser | ✅ Done | Medium | - |
| poodle parser | ✅ Done | Low | - |
| universalmutator parser | ✅ Done | Medium | - |

**Exit Criteria**: Run any Python mutation tool through single MCP interface — **MET**

---

### M2: Graph-Spectral Infrastructure (Q1-Q2)
**Goal**: O(1) reachability queries via hierarchical decomposition

| Task | Status | Complexity | Dependencies |
|------|--------|------------|--------------|
| Graph models (CodeGraph, Node, Edge) | ✅ Done | Low | - |
| Call graph extraction (AST) | ✅ Done | Medium | - |
| ASR extractor (import/class analysis) | ✅ Done | Medium | - |
| Dependency graph extraction | ✅ Done | Medium | - |
| Laplacian construction | ✅ Done | Low | Graph extraction |
| Fiedler vector computation | ✅ Done | Medium | Laplacian |
| Recursive partitioning | ✅ Done | Medium | Fiedler |
| SCC detection + condensation | ✅ Done | Low | Partitioning |
| Planar subgraph identification | ✅ Done | High | SCC |
| Kameda preprocessing | ✅ Done | High | Planar subgraph |
| Virtual sink/source augmentation | ✅ Done | Low | Module detection |
| MCP tools for graph queries | ✅ Done | Low | All above |

**Exit Criteria**: Query reachability between any two functions in O(1) after O(n) preprocessing — **MET**

---

### M3: Belief Revision Engine (Q2)
**Goal**: AGM-compliant theory management with provenance

| Task | Status | Complexity | Dependencies |
|------|--------|------------|--------------|
| py-brs library (AGM core) | ✅ Done | High | - |
| Evidence adapter (mutation→belief) | ✅ Done | Medium | py-brs |
| Theory manager (curate-ipsum) | ✅ Done | Medium | Evidence adapter |
| AGM contraction (py-brs v2.0.0) | ✅ Done | High | py-brs |
| Assertion model (types, behaviors) | ✅ Done | Medium | - |
| Entrenchment calculation (py-brs v2.0.0) | ✅ Done | Medium | Evidence |
| Provenance DAG storage | ✅ Done | Medium | AGM operations |
| Rollback mechanism | ✅ Done | Medium | Provenance DAG |
| Failure mode analyzer | ✅ Done | High | All above |

**Exit Criteria**: Track belief evolution across synthesis attempts with full provenance — **MET**

---

### M4: Synthesis Loop (Q2-Q3)
**Goal**: LLM candidates → verified patches

| Task | Complexity | Dependencies |
|------|------------|--------------|
| LLM candidate extraction (top-k) | Low | - |
| Population initialization | Low | LLM extraction |
| Fitness function (CE avoidance + spec) | Medium | M3 |
| AST-aware crossover | High | Population |
| Directed mutation (CE-guided) | High | Fitness |
| Entropy monitoring | Medium | Population |
| Diversity injection | Medium | Entropy |
| CEGIS main loop | High | All above |

**Exit Criteria**: Generate patch that kills previously-surviving mutant, verified correct

---

### M5: Verification Backends (Q3)
**Goal**: Formal verification infrastructure

| Task | Complexity | Dependencies |
|------|------------|--------------|
| Z3 Python bindings integration | Low | - |
| Type abstraction level (CEGAR) | Medium | - |
| CFG abstraction level | Medium | Type level |
| DFG abstraction level | High | CFG level |
| Concrete execution level | Medium | DFG level |
| Spurious CE detection | High | All levels |
| SymPy path condition encoding | Medium | - |
| Numerical solver fallback | Medium | SymPy |
| KLEE container integration | High | - |

**Exit Criteria**: Verify patch correctness against specification with proof certificate

---

### M6: Graph Database + RAG (Q3-Q4)
**Goal**: Persistent, queryable code graph

| Task | Complexity | Dependencies |
|------|------------|--------------|
| Joern CPG generation | Medium | - |
| Neo4j schema design | Medium | CPG |
| Reachability index persistence | Medium | M2, Neo4j |
| Incremental update on file change | High | Index |
| Code embedding model | Medium | - |
| Semantic search index | Medium | Embedding |
| RAG retrieval pipeline | Medium | Search index |
| MCP tools for graph queries | Low | All above |

**Exit Criteria**: Natural language queries over codebase with graph-backed retrieval

---

### M7: Production Hardening (Q4)
**Goal**: CI/CD-ready deployment

| Task | Complexity | Dependencies |
|------|------------|--------------|
| GitHub Actions integration | Low | M1 |
| Regression detection (PID d-term) | Low | M1 |
| Threshold-based quality gates | Low | Regression |
| HTML report generation | Medium | - |
| SARIF output format | Low | - |
| VSCode extension | High | MCP interface |
| Self-healing metadata consistency | High | M2, M3 |
| Performance benchmarking | Medium | All |

**Exit Criteria**: Drop-in CI integration with automated quality gates

---

## Critical Path

```
M1 (Frameworks) ──→ M2 (Graph) ──→ M6 (Graph DB)
       │                │
       ▼                ▼
M3 (Belief) ←────→ M4 (Synthesis)
       │                │
       ▼                ▼
M5 (Verification) ─────→ M7 (Production)
```

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fiedler computation slow for large graphs | M2 delay | Sparse eigensolvers, approximate methods |
| Planar subgraph identification NP-hard | M2 delay | Heuristic identification, accept suboptimality |
| CEGIS loop non-convergence | M4 failure | Entropy injection, timeout with best-effort |
| Z3 timeout on complex constraints | M5 delay | SymPy reformulation, numerical fallback |
| LLM API rate limits | M4 slowdown | Local model fallback (CodeLlama) |

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Mutation score improvement | +15% | Before/after synthesis |
| Patch verification rate | >80% | Patches that pass CEGAR |
| Reachability query time | <1ms | p99 latency |
| False positive rate | <5% | Spurious CE ratio |
| Time to verified patch | <5min | End-to-end for single mutant |

## Resource Requirements

### Compute
- Development: Standard workstation
- CI: 4-core runners with 16GB RAM
- KLEE/Z3: Dedicated container with 32GB+ RAM
- Graph DB: Neo4j instance (can start with embedded)

### Dependencies
- Python 3.10+
- scipy, networkx (graph algorithms)
- z3-solver (SMT)
- sympy (symbolic math)
- pydantic (models)
- FastMCP (server)

### Optional
- KLEE (concolic execution)
- Neo4j/JanusGraph (graph persistence)
- Joern (CPG generation)
