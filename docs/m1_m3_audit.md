# Milestones M1-M3 Completeness Audit

**Date**: 2026-01-27
**Auditor**: Claude
**Project**: curate-ipsum

## Executive Summary

| Milestone | Status | Completion | Notes |
|-----------|--------|------------|-------|
| **M1**: Multi-Framework Foundation | ⚠️ PARTIAL | ~50% | Core parsers done, 3 frameworks missing |
| **M2**: Graph-Spectral Infrastructure | ❌ NOT STARTED | 0% | No implementation |
| **M3**: Belief Revision Engine | ✅ MOSTLY DONE | ~85% | Via py-brs integration |

---

## M1: Multi-Framework Foundation

**Goal**: Unified interface across mutation testing tools
**Exit Criteria**: Run any Python mutation tool through single MCP interface

### Task Breakdown

| Task | Complexity | Status | Evidence |
|------|------------|--------|----------|
| mutmut parser | Low | ✅ DONE | `parsers/mutmut_parser.py` (340 lines) |
| cosmic-ray parser | Medium | ❌ NOT DONE | Only error stub in `parsers/__init__.py:131-135` |
| poodle parser | Low | ❌ NOT DONE | Not implemented |
| universalmutator parser | Medium | ❌ NOT DONE | Not implemented |
| Framework auto-detection | Low | ✅ DONE | `parsers/detection.py` (354 lines) |
| Non-contradictory region assignment | Medium | ⚠️ PARTIAL | Region model done, assignment logic missing |

### Detailed Analysis

#### ✅ mutmut parser
- Full SQLite cache parser supporting v1 and v2 schemas
- Status mapping: `ok_killed`, `bad_survived`, `bad_timeout`, `ok_suspicious`, `untested`, `skipped`
- Region-level filtering via `get_mutmut_region_mutants()`
- Comprehensive test coverage in `tests/test_m1_parsers.py`

#### ✅ Framework auto-detection
- Language detection from file extensions and config files
- Framework detection from output files, cache, and config
- Recommendation engine based on language and existing setup
- MCP tool: `detect_frameworks_tool()`

#### ⚠️ Non-contradictory region assignment
**Implemented**:
- `Region` model with hierarchical levels (FILE, CLASS, FUNCTION, LINES)
- `contains()` and `overlaps()` methods for relationship checking
- String serialization: `file:path::class:name::func:name::lines:start-end`
- MCP tools: `parse_region_tool()`, `check_region_relationship_tool()`, `create_region_tool()`

**Missing**:
- Automatic region assignment from mutation results
- Region hierarchy building from AST
- Non-contradictory assignment algorithm (ensuring nested regions don't conflict)

#### ❌ cosmic-ray parser
Current state: Error stub only
```python
elif tool_lower in ("cosmic_ray", "cosmicray", "cosmic"):
    raise UnsupportedFrameworkError(
        f"cosmic-ray parser not yet implemented. "
        f"Supported frameworks: stryker, mutmut"
    )
```

**Required**:
- Parse cosmic-ray's SQLite session database
- Status mapping from cosmic-ray's outcomes (KILLED, SURVIVED, INCOMPETENT, etc.)
- Integration with unified parser interface

#### ❌ poodle / universalmutator parsers
No implementation started. Would need:
- poodle: Parse JSON/text output format
- universalmutator: Parse output format (varies by target language)

### Exit Criteria Assessment

**"Run any Python mutation tool through single MCP interface"**

| Tool | Supported | Notes |
|------|-----------|-------|
| mutmut | ✅ Yes | Full support |
| cosmic-ray | ❌ No | UnsupportedFrameworkError |
| poodle | ❌ No | UnsupportedFrameworkError |
| universalmutator | ❌ No | UnsupportedFrameworkError |
| Stryker (JS/TS) | ✅ Yes | Full support |

**Exit criteria NOT MET** - only 2/5 Python-relevant tools supported.

---

## M2: Graph-Spectral Infrastructure

**Goal**: O(1) reachability queries via hierarchical decomposition
**Exit Criteria**: Query reachability between any two functions in O(1) after O(n) preprocessing

### Task Breakdown

| Task | Complexity | Status | Evidence |
|------|------------|--------|----------|
| Call graph extraction (AST) | Medium | ❌ NOT DONE | - |
| Dependency graph extraction | Medium | ❌ NOT DONE | - |
| Laplacian construction | Low | ❌ NOT DONE | - |
| Fiedler vector computation | Medium | ❌ NOT DONE | - |
| Recursive partitioning | Medium | ❌ NOT DONE | - |
| SCC detection + condensation | Low | ❌ NOT DONE | - |
| Planar subgraph identification | High | ❌ NOT DONE | - |
| Kameda preprocessing | High | ❌ NOT DONE | - |
| Virtual sink/source augmentation | Low | ❌ NOT DONE | - |

### Detailed Analysis

**No implementation exists.** This milestone has not been started.

The infrastructure would require:
1. AST parsing for Python/JS to extract function call relationships
2. NetworkX or similar for graph operations
3. SciPy for sparse eigensolvers (Fiedler vector)
4. Custom algorithms for Kameda preprocessing (planar reachability)

### Exit Criteria Assessment

**"Query reachability between any two functions in O(1) after O(n) preprocessing"**

**Exit criteria NOT MET** - 0% implementation.

---

## M3: Belief Revision Engine

**Goal**: AGM-compliant theory management with provenance
**Exit Criteria**: Track belief evolution across synthesis attempts with full provenance

### Task Breakdown

| Task | Complexity | Status | Evidence |
|------|------------|--------|----------|
| Assertion model | Medium | ✅ DONE | `theory/manager.py:add_assertion()` |
| Evidence types + grounding rules | Low | ✅ DONE | `adapters/evidence_adapter.py` |
| Entrenchment calculation | Medium | ✅ DONE | `theory/manager.py:get_entrenchment()` |
| AGM expansion/contraction/revision | High | ✅ DONE | Via py-brs `contract()`, `revise()` |
| Provenance DAG storage | Medium | ✅ DONE | Via py-brs CASStore WorldBundle |
| Rollback mechanism | Medium | ⚠️ PARTIAL | World forking exists, no explicit API |
| Failure mode analyzer | High | ❌ NOT DONE | - |

### Detailed Analysis

#### ✅ Assertion model
Supports typed assertions:
- `type` - Type assertions about code
- `behavior` - Behavioral properties
- `invariant` - Invariant conditions
- `contract` - Pre/post conditions
- `precondition`, `postcondition`

Each assertion has:
- Confidence score (0.0-1.0)
- Region binding (optional)
- Evidence grounding (required)

#### ✅ Evidence types + grounding rules
- `test_result_to_evidence()` - Maps TestRunResult → BRS Evidence
- `mutation_result_to_evidence()` - Maps MutationRunResult → BRS Evidence
- Evidence reliability derived from test outcomes
- Grounding edges link Evidence → Assertion nodes

#### ✅ Entrenchment calculation
Via py-brs `compute_entrenchment()`:
- Considers incoming edge tiers
- Weighs by confidence scores
- Returns 0.0-1.0 resilience score

#### ✅ AGM expansion/contraction/revision
- **Expansion**: `add_assertion()` - Adds new belief with evidence
- **Contraction**: `contract_assertion()` - Removes belief via AGM contraction
  - Strategies: `entrenchment`, `minimal`, `full_cascade`
- **Revision**: `revise_with_assertion()` - Implements Levi identity K*φ = (K÷¬φ)+φ

#### ✅ Provenance DAG storage
Via py-brs CASStore:
- Content-addressed storage (immutable objects)
- WorldBundle tracks: node_ids, edge_ids, evidence_ids
- Version labels for world states
- Full audit trail via hash chains

#### ⚠️ Rollback mechanism
**Implemented**:
- World forking via `store.get_world(domain, label)`
- Can create new world versions
- Can query historical worlds by hash

**Missing**:
- Explicit `rollback_to(version)` API
- World comparison / diff tools
- Time-travel queries

#### ❌ Failure mode analyzer
No implementation. Would analyze:
- Why beliefs were contracted
- Contradiction patterns
- Evidence invalidation chains
- Synthesis attempt failure modes

### MCP Tools Implemented

| Tool | Function |
|------|----------|
| `add_assertion_tool` | Add typed assertion with evidence |
| `contract_assertion_tool` | AGM contraction |
| `get_entrenchment_tool` | Query entrenchment score |
| `list_assertions_tool` | List/filter assertions |
| `get_theory_snapshot_tool` | World state snapshot |
| `revise_theory_tool` | AGM revision |

### Exit Criteria Assessment

**"Track belief evolution across synthesis attempts with full provenance"**

| Capability | Status |
|------------|--------|
| Track belief additions | ✅ Yes |
| Track belief removals | ✅ Yes |
| Track belief revisions | ✅ Yes |
| Query provenance | ✅ Yes (via CASStore) |
| Analyze failure modes | ❌ No |

**Exit criteria MOSTLY MET** - core belief tracking works, failure analysis missing.

---

## Recommendations

### Priority 1: Complete M1 Exit Criteria
1. Implement cosmic-ray parser (most common after mutmut)
2. Defer poodle/universalmutator (lower adoption)
3. Add region auto-assignment from AST

### Priority 2: M3 Gap Closure
1. Add explicit `rollback_to(version)` API
2. Implement basic failure mode analyzer

### Priority 3: M2 (defer or reduce scope)
M2 is a significant undertaking. Consider:
- Using existing tools (e.g., `pyan` for Python call graphs)
- Reducing scope to basic reachability (no O(1) requirement)
- Deferring until M4 (Synthesis Loop) actually needs it

---

## Test Coverage

| Module | Test File | Coverage |
|--------|-----------|----------|
| regions | `tests/test_m1_regions.py` | Good (356 lines) |
| parsers | `tests/test_m1_parsers.py` | Good (458 lines) |
| BRS integration | `tests/test_brs_integration.py` | Basic |
| theory | - | Needs tests |
| adapters | - | Needs tests |

---

## Files Inventory

### M1 Implementation
```
parsers/
├── __init__.py         # Unified interface (169 lines)
├── detection.py        # Language/framework detection (354 lines)
├── stryker_parser.py   # Stryker JSON parser (236 lines)
└── mutmut_parser.py    # mutmut SQLite parser (434 lines)

regions/
├── __init__.py         # Module exports
└── models.py           # Region model (325 lines)
```

### M3 Implementation
```
theory/
├── __init__.py         # Module exports
└── manager.py          # TheoryManager (486 lines)

adapters/
├── __init__.py         # Module exports
└── evidence_adapter.py # Evidence mapping

domains/
├── __init__.py         # Module exports
└── code_mutation_smoke.py  # BRS domain extension
```

### M2 Implementation
```
(none)
```
