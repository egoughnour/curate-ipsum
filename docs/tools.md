# MCP Tool Reference

Curate-Ipsum exposes **30 tools** over the MCP stdio transport, organised into
six groups.

## Testing

Tools for running tests, detecting frameworks, and tracking metrics.

### `run_unit_tests`

Run unit tests for a project and return a summarised result.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `projectId` | string | yes | Project identifier |
| `commitSha` | string | yes | Current commit SHA |
| `command` | string | yes | Shell command to execute |
| `workingDirectory` | string | yes | Project root directory |
| `regionId` | string | no | Scope to a specific region |
| `framework` | string | no | Test framework name (default: `generic`) |

### `run_integration_tests`

Same parameters as `run_unit_tests`. Runs integration tests separately so
metrics are tracked independently.

### `run_mutation_tests`

Run mutation tests and return summarised mutation statistics.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `projectId` | string | yes | Project identifier |
| `commitSha` | string | yes | Current commit SHA |
| `command` | string | yes | Mutation tool command |
| `workingDirectory` | string | yes | Project root directory |
| `regionId` | string | no | Scope to a specific region |
| `tool` | string | no | Framework override (auto-detected if omitted) |
| `reportPath` | string | no | Custom report file path |

Supported frameworks: Stryker (JS/TS), mutmut (Python), cosmic-ray, poodle,
universalmutator.

### `get_run_history`

Return recent test and mutation runs for a project.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `projectId` | string | yes | Project identifier |
| `regionId` | string | no | Filter to a region |
| `limit` | int | no | Max results to return |

### `get_region_metrics`

Compute PID-like metrics and mutation score for a specific code region.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `projectId` | string | yes | Project identifier |
| `commitSha` | string | yes | Current commit SHA |
| `regionId` | string | yes | Region identifier |

### `detect_frameworks`

Detect available mutation testing frameworks and project language.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `workingDirectory` | string | yes | Project root directory |

### `parse_region`, `check_region_relationship`, `create_region`

Utilities for working with the hierarchical region model
(`file:path::class:Name::func:name::lines:1-10`).

---

## Belief Revision

AGM-compliant belief revision powered by [py-brs](https://pypi.org/project/py-brs/).

### `add_assertion`

Add a typed assertion grounded by evidence.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `projectId` | string | yes | Project identifier |
| `assertionType` | string | yes | `type`, `behavior`, `invariant`, or `contract` |
| `content` | string | yes | Assertion statement |
| `evidenceId` | string | yes | Grounding evidence ID |
| `confidence` | float | no | Confidence 0.0–1.0 (default: 0.5) |
| `regionId` | string | no | Scope to a region |

### `contract_assertion`

Remove an assertion via AGM contraction.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `projectId` | string | yes | Project identifier |
| `nodeId` | string | yes | Assertion node to contract |
| `strategy` | string | no | `entrenchment`, `minimal`, or `full_cascade` |

### `revise_theory`

Incorporate a new assertion, contracting contradictions first (Levi identity).

### `get_entrenchment`

Get the entrenchment score (0.0–1.0) for an assertion — how resilient it is
to contraction.

### `list_assertions`, `get_theory_snapshot`

Query the current theory state.

### `store_evidence`, `get_provenance`, `why_believe`, `belief_stability`

Evidence management and provenance tracing.

---

## Rollback & Failure Analysis

### `rollback_to`

Revert the theory to a prior world state by content-addressable hash.

### `undo_last_operations`

Walk backward N operations through the provenance DAG.

### `analyze_failure`

Classify a synthesis failure (type mismatch, overfitting, underfitting, etc.)
and suggest which assertions to contract.

### `list_world_history`

List all historical world states with timestamps and reasons.

---

## Graph-Spectral Analysis

Call graph extraction, Fiedler spectral partitioning, Kameda O(1) reachability,
and hierarchical decomposition.

### `extract_call_graph`

Extract and analyse a Python project's call graph. Returns node count, edge
count, SCCs, connected components, and function list. Persists the graph to
the configured store.

### `compute_partitioning`

Fiedler spectral partitioning — recursively bipartitions the call graph using
the second eigenvector of the Laplacian. Returns a partition tree with
algebraic connectivity values.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `workingDirectory` | string | yes | Project root |
| `min_partition_size` | int | no | Minimum partition size (default: 3) |
| `max_depth` | int | no | Maximum recursion depth (default: 10) |

### `query_reachability`

O(1) reachability check between two functions using the Kameda index on planar
subgraphs with BFS fallback for non-planar edges.

### `get_hierarchy`

Hierarchical decomposition alternating SCC condensation and Fiedler partitioning.

### `find_function_partition`

Locate which partition a function belongs to, its siblings, and the partition's
entry/exit points.

### `incremental_update`, `persistent_graph_stats`, `graph_query`

Persistent graph store operations: incremental re-extraction, statistics, and
structured queries (neighbors, reachability, node lookup).

---

## Verification

Formal verification via Z3 SMT solving and angr Docker symbolic execution.

### `verify_property`

Run formal verification on a constraint set.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `backend` | string | no | `z3` (default), `angr`, or `mock` |
| `constraints` | list[str] | no | Z3 constraints in mini-DSL |
| `timeoutSeconds` | int | no | Budget timeout (default: 30) |
| `maxStates` | int | no | angr state limit (default: 50000) |

### `verify_with_orchestrator`

CEGAR orchestrator with budget escalation (10s → 30s → 120s). Chains
verification attempts with progressively larger budgets.

### `list_verification_backends`

Enumerate available backends and their capabilities.

---

## Synthesis & RAG

CEGIS synthesis loop with LLM seeding, genetic algorithm evolution, and
RAG-augmented context retrieval.

### `synthesize_patch`

The main synthesis entry point. Generates a verified patch to kill a surviving
mutant.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `projectId` | string | yes | Project identifier |
| `workingDirectory` | string | yes | Project root |
| `testCommand` | string | yes | How to run tests |
| `llmBackend` | string | no | `mock`, `cloud`, or `local` |
| `maxIterations` | int | no | CEGIS budget (default: 50) |
| `populationSize` | int | no | Genetic algorithm population (default: 20) |

### `synthesis_status`, `cancel_synthesis`, `list_synthesis_runs`

Monitor and manage synthesis runs.

### `rag_index_nodes`

Index code nodes into ChromaDB for semantic search.

### `rag_search`

Semantic search with optional call-graph expansion (vector top-k → neighbor
expansion → rerank).

### `rag_stats`

Vector store statistics.
