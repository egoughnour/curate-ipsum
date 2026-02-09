# Architecture

Curate-Ipsum is structured as a layered system with six major subsystems,
all exposed through a single MCP server.

## System overview

```
LLM Candidates (k samples)
        |
   Seed Population
        |
+---------------------------+
|  CEGIS + CEGAR + Genetic  |  <-- Verification loop
|  + Belief Revision        |
+---------------------------+
        |
  Strongly Typed Patch
  (with proof certificate)
```

## Subsystems

### Parsers & Detection

Multi-framework mutation testing orchestration. Supports Stryker (JS/TS),
mutmut (Python), cosmic-ray, poodle, and universalmutator. Auto-detects
project language and recommends a framework.

### Graph-Spectral Analysis

Call graph extraction, Fiedler spectral partitioning, Kameda O(1) reachability
queries, and hierarchical SCC+Fiedler decomposition.

Key algorithms:

- **Fiedler vector**: Second eigenvector of the graph Laplacian — optimal
  bipartition criterion.
- **Kameda index**: O(1) reachability queries after O(n) preprocessing on
  planar subgraphs.
- **Boyer-Myrvold**: Planarity testing with Kuratowski subgraph extraction
  for non-planar edge identification.

### Belief Revision Engine

AGM-compliant theory management powered by
[py-brs](https://pypi.org/project/py-brs/). Supports expansion, contraction
(minimal, entrenchment, full cascade), and revision (Levi identity).
Provenance DAG tracks all belief revision operations.

### Synthesis Loop

CEGIS (Counterexample-Guided Inductive Synthesis) with:

- LLM-seeded initial population
- Genetic algorithm with AST-aware crossover/mutation
- Entropy monitoring and diversity injection
- Fitness function incorporating test pass rate, mutation score, and type
  correctness
- RAG-augmented context retrieval for better LLM prompts

### Verification Backends

Two-tier verification:

- **Z3** (default, cheap): SMT constraint solving with mini-DSL parser
- **angr** (expensive): Docker-containerised symbolic execution with JSON
  file exchange

CEGAR orchestrator escalates budgets: 10s → 30s → 120s.

### RAG / Semantic Search

ChromaDB vector store with `all-MiniLM-L6-v2` embeddings. The search pipeline
does vector top-k retrieval, then expands results using the call graph
(callees + callers), reranks, and packs context for synthesis prompts.

## Persistence

All persistent state lives under `MUTATION_TOOL_DATA_DIR`:

- **Run history**: JSONL append log
- **Theory state**: Content-addressable worlds (rollback-safe)
- **Graph store**: SQLite (or Kuzu) with nodes, edges, reachability index
- **Synthesis results**: JSON per run
- **Vector store**: ChromaDB (ephemeral in-process or Docker server)

## Design documents

See the [design documents](design/index.md) section for the original
architectural plans, decision logs, and framework analyses.
