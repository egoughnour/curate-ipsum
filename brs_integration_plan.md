# BRS Integration Plan for Curate-Ipsum

## Executive Summary

The `py-brs` (Belief Revision System) package provides **substantial infrastructure** that directly addresses Phase 4 (Belief Revision Engine) of the curate-ipsum roadmap, with significant applicability to Phases 3, 5, and 7. This integration can accelerate development by 2-3 months by reusing battle-tested components.

## BRS Feature Inventory

| Component | What It Provides | Status |
|-----------|------------------|--------|
| `core.py` | Node, Edge, Evidence, Pattern, WorldBundle, Maturity, Proposal | ✅ Production-ready |
| `storage.py` | CASStore (SQLite + content-addressable) | ✅ Production-ready |
| `revision.py` | Shadow import, evaluate_proposal, world forking | ✅ Production-ready |
| `inference.py` | Graph traversal, Dijkstra path-finding, ancestry | ✅ Production-ready |
| `mesh.py` | Pattern signatures, similarity metrics, cross-domain matching | ✅ Production-ready |
| `discovery.py` | Analog suggestion, proposal persistence | ✅ Production-ready |
| `evaluator.py` | Smoke test orchestration, batch evaluation | ✅ Production-ready |
| `domains/registry.py` | Pluggable domain system with auto-discovery | ✅ Production-ready |
| `cli.py` | Command-line interface | ✅ Production-ready |

## Roadmap Coverage Analysis

### Phase 4: Belief Revision Engine ✅ **85% COVERED**

| Roadmap Item | BRS Coverage | Notes |
|--------------|--------------|-------|
| AGM-compliant theory representation | ✅ `WorldBundle` + versioning | Immutable snapshots with domain/version |
| Evidence types and grounding rules | ✅ `Evidence` class | reliability (A/B/C), citation, kind |
| Entrenchment calculation | ✅ `Edge.tier` + `confidence` | 6-level tiering, 0.0-1.0 confidence |
| Contraction via minimal hitting sets | ⚠️ Partial | Implicit via rejection; needs explicit contraction |
| Provenance DAG storage and queries | ✅ `CASStore` + hash chains | Content-addressable, full history |

**Gap**: BRS has implicit contraction (reject proposal → world unchanged) but no explicit AGM contraction operation. Need to add `contract()` method.

### Phase 3: Multi-Framework Orchestration ⚠️ **40% COVERED**

| Roadmap Item | BRS Coverage | Notes |
|--------------|--------------|-------|
| Unified mutation framework interface | ❌ Not present | Need new adapter layer |
| Implicit region detection | ✅ `Pattern` + `mesh.py` | Cross-domain pattern matching |
| Non-contradictory framework assignment | ⚠️ Partial | Proposal status tracking exists |
| Cross-framework survival analysis | ⚠️ Partial | `Maturity` scoring applicable |

**Gap**: BRS doesn't know about mutation testing tools. Need domain extension for "code_mutation" domain.

### Phase 5: Synthesis Loop ⚠️ **30% COVERED**

| Roadmap Item | BRS Coverage | Notes |
|--------------|--------------|-------|
| CEGIS implementation | ❌ Not present | Need new module |
| CEGAR abstraction hierarchy | ❌ Not present | Need new module |
| Genetic algorithm | ❌ Not present | Need new module |
| Counterexample handling | ✅ `evaluate_proposal` | Shadow testing, maturity delta |
| Population management | ⚠️ Partial | `Proposal` with status tracking |

**Gap**: Core synthesis algorithms not in BRS. BRS provides the belief management infrastructure that the synthesis loop would use.

### Phase 6: Verification Backends ❌ **0% COVERED**

| Roadmap Item | BRS Coverage | Notes |
|--------------|--------------|-------|
| Z3 integration | ❌ Not present | Need new module |
| KLEE container | ❌ Not present | Need new module |
| SymPy encoding | ❌ Not present | Need new module |

**Gap**: BRS is domain-agnostic and doesn't include verification backends. These are orthogonal.

### Phase 7: Graph Database Integration ⚠️ **50% COVERED**

| Roadmap Item | BRS Coverage | Notes |
|--------------|--------------|-------|
| Graph storage | ✅ `CASStore` (SQLite) | Works but not Neo4j/Joern |
| Reachability queries | ✅ `inference.py` | Dijkstra, BFS, ancestry |
| Incremental update | ✅ World forking | Non-destructive updates |
| Semantic search | ✅ `mesh.py` | Pattern similarity |

**Gap**: BRS uses SQLite, not Neo4j/Joern. Graph algorithms are pure Python, not leveraging graph DB query optimization.

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Curate-Ipsum MCP Interface                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    NEW: Synthesis Layer                         │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │ │
│  │  │  CEGIS   │  │  CEGAR   │  │ Genetic  │  │ LLM Seed │       │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │ │
│  │       └─────────────┴─────────────┴─────────────┘              │ │
│  └──────────────────────────────┬─────────────────────────────────┘ │
│                                 │                                    │
│  ┌──────────────────────────────┴─────────────────────────────────┐ │
│  │                    REUSE: py-brs Core                           │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │ │
│  │  │ revision │  │ storage  │  │ inference│  │   mesh   │       │ │
│  │  │ (shadow) │  │ (CAS)    │  │ (graph)  │  │ (analog) │       │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │ │
│  │  │evaluator │  │ discovery│  │  core    │                     │ │
│  │  │ (smoke)  │  │ (propose)│  │ (types)  │                     │ │
│  │  └──────────┘  └──────────┘  └──────────┘                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                 │                                    │
│  ┌──────────────────────────────┴─────────────────────────────────┐ │
│  │                    NEW: Domain Extensions                       │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │ │
│  │  │code_mutation │  │code_coverage │  │ code_graph   │         │ │
│  │  │   _smoke.py  │  │   _smoke.py  │  │  _smoke.py   │         │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                 │                                    │
│  ┌──────────────────────────────┴─────────────────────────────────┐ │
│  │                    NEW: Verification Layer                      │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │ │
│  │  │    Z3    │  │   KLEE   │  │  SymPy   │                     │ │
│  │  └──────────┘  └──────────┘  └──────────┘                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Concrete Mapping: BRS → Curate-Ipsum Concepts

### Type Mappings

| Curate-Ipsum Concept | BRS Type | Adaptation Needed |
|---------------------|----------|-------------------|
| `Assertion` | `Node` | Add `assertion_type` to properties |
| `Evidence` | `Evidence` | Compatible as-is |
| `SynthesisTheory` | `WorldBundle` | Add synthesis-specific metadata |
| `TypedPatch` | `Proposal` | Add code diff fields |
| `Entrenchment` | `Edge.tier` + `Edge.confidence` | May need finer granularity |
| `RunResult` | `Evidence` | Map mutation run → evidence |
| `RegionMetrics` | `Maturity` | Add centrality/triviality |

### Code Domain Extension

```python
# brs/domains/code_mutation_smoke.py

DOMAIN_ID = "code_mutation"

def run(store, domain_id, world_label):
    """Smoke tests for code mutation domain."""
    tests, failures, messages = 0, 0, []

    # Test 1: All assertions have evidence
    tests += 1
    # ... check evidence grounding

    # Test 2: No contradictory type assertions
    tests += 1
    # ... check type consistency

    # Test 3: Mutation score monotonicity
    tests += 1
    # ... check historical improvement

    return (tests, failures, messages)

SMOKE_LEVELS = {
    "smoke": run,
    "regression": run_regression,
    "deep": run_deep_verification,
}
```

### Evidence Type Extensions

```python
# Extend Evidence.kind for code synthesis

class CodeEvidenceKind(str, Enum):
    TEST_PASS = "test_pass"
    TEST_FAIL = "test_fail"
    MUTATION_KILLED = "mutation_killed"
    MUTATION_SURVIVED = "mutation_survived"
    TYPE_CHECK_PASS = "type_check_pass"
    TYPE_CHECK_FAIL = "type_check_fail"
    SMT_SAT = "smt_sat"
    SMT_UNSAT = "smt_unsat"
    COUNTEREXAMPLE = "counterexample"
    PROOF_CERTIFICATE = "proof_certificate"
    LLM_SUGGESTION = "llm_suggestion"  # Low reliability
```

### Reliability Mapping

| Evidence Type | BRS Reliability | Rationale |
|---------------|-----------------|-----------|
| PROOF_CERTIFICATE | A | Formal proof |
| SMT_UNSAT | A | Mathematical certainty |
| SMT_SAT | A | Concrete witness |
| COUNTEREXAMPLE | A | Concrete falsification |
| TYPE_CHECK_* | B | Static analysis |
| TEST_PASS/FAIL | B | Dynamic but incomplete |
| MUTATION_* | B | Heuristic quality signal |
| LLM_SUGGESTION | C | Statistical plausibility only |

## Implementation Plan

### Step 1: Add py-brs as Dependency (Day 1)

```toml
# pyproject.toml
dependencies = [
    "py-brs>=1.0.0",
    "pydantic>=2.0",
    # ... existing deps
]
```

### Step 2: Create Code Mutation Domain (Days 2-3)

```python
# curate_ipsum/domains/code_mutation.py

from brs import Node, Edge, Evidence, Pattern, WorldBundle
from brs.domains.registry import register_smoke_test

DOMAIN_ID = "code_mutation"

# Node types for code entities
class CodeNodeType(str, Enum):
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    REGION = "region"
    ASSERTION = "assertion"

# Pattern types for code invariants
class CodePatternType(str, Enum):
    TYPE_SIGNATURE = "type_signature"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    BEHAVIOR = "behavior"
```

### Step 3: Integrate with Existing Tools (Days 4-7)

```python
# curate_ipsum/adapters/mutation_to_brs.py

from brs import CASStore, Evidence
from curate_ipsum.tools import MutationRunResult

def mutation_result_to_evidence(run: MutationRunResult) -> Evidence:
    """Convert mutation run to BRS evidence."""
    return Evidence(
        id=f"mutation_{run.id}",
        citation=f"Mutation run {run.id} at {run.timestamp}",
        kind="mutation_killed" if run.killed > 0 else "mutation_survived",
        reliability="B",
        date=run.timestamp.isoformat(),
        metadata={
            "tool": run.tool,
            "killed": run.killed,
            "survived": run.survived,
            "score": run.mutationScore,
        }
    )

def test_result_to_evidence(run: TestRunResult) -> Evidence:
    """Convert test run to BRS evidence."""
    return Evidence(
        id=f"test_{run.id}",
        citation=f"Test run {run.id} at {run.timestamp}",
        kind="test_pass" if run.passed else "test_fail",
        reliability="B",
        date=run.timestamp.isoformat(),
        metadata={
            "framework": run.framework,
            "passed": run.passedTests,
            "failed": run.failedTests,
        }
    )
```

### Step 4: Synthesis Loop Using BRS (Days 8-14)

```python
# curate_ipsum/synthesis/cegis.py

from brs import CASStore, Proposal, WorldBundle
from brs.revision import evaluate_proposal, shadow_import_analog
from brs.evaluator import evaluate_with_registry

class CEGISEngine:
    def __init__(self, store: CASStore):
        self.store = store

    def synthesize(
        self,
        spec: Specification,
        llm_candidates: List[CodePatch],
    ) -> Optional[StronglyTypedPatch]:

        # Create proposals from LLM candidates
        for candidate in llm_candidates:
            proposal = self._candidate_to_proposal(candidate)
            self.store.put_proposal("code_patch", proposal, "new")

        # CEGIS loop
        counterexamples = []
        for iteration in range(MAX_ITERATIONS):
            # Get best proposal
            proposals = self.store.list_proposals(status="new")
            best = self._select_best(proposals, counterexamples)

            # Evaluate via BRS shadow testing
            result = evaluate_with_registry(
                self.store,
                best.id,
                target_domain="code_mutation",
                config=EvaluationConfig(
                    base_world="green",
                    shadow_prefix="_cegis_",
                    acceptance_threshold=0.1,
                )
            )

            if result.outcome == "accepted":
                return self._to_strongly_typed(best, result)

            # Extract counterexample and refine
            ce = self._extract_counterexample(result)
            counterexamples.append(ce)

            # Update proposal statuses
            self._refine_proposals(counterexamples)

        return None
```

### Step 5: Wire Up MCP Tools (Days 15-17)

```python
# curate_ipsum/server.py

from brs import CASStore
from curate_ipsum.synthesis.cegis import CEGISEngine

@server.tool(description="Synthesize verified patch from LLM candidates")
async def synthesize_patch(
    projectId: str,
    regionId: str,
    specification: str,
    llm_candidates: List[str],
) -> dict:
    store = CASStore(Path(DATA_DIR) / projectId)
    engine = CEGISEngine(store)

    result = engine.synthesize(
        spec=parse_spec(specification),
        llm_candidates=[CodePatch(code=c) for c in llm_candidates],
    )

    if result:
        return result.model_dump()
    return {"error": "Synthesis failed"}
```

## Revised Roadmap with BRS

### Phase 1: Foundation (Current) - No Change

### Phase 2: Graph Infrastructure - Minor Savings
- BRS `inference.py` provides graph traversal
- Still need Fiedler/Kameda (not in BRS)

### Phase 3: Multi-Framework Orchestration - 2 Weeks Saved
- Reuse `Pattern` for implicit regions
- Reuse `mesh.py` for cross-framework analysis
- Still need mutation tool adapters

### Phase 4: Belief Revision Engine - **6 Weeks Saved** ✅
- Reuse `CASStore` for storage
- Reuse `WorldBundle` for theory snapshots
- Reuse `Evidence` for grounding
- Reuse `revision.py` for shadow evaluation
- Reuse `evaluator.py` for smoke tests
- **Only need**: Explicit contraction, synthesis-specific extensions

### Phase 5: Synthesis Loop - 1 Week Saved
- Reuse `Proposal` for candidate management
- Reuse evaluation infrastructure
- **Still need**: CEGIS, CEGAR, genetic algorithms

### Phase 6: Verification Backends - No Change
- BRS doesn't cover this

### Phase 7: Graph Database - 1 Week Saved
- Reuse `CASStore` as baseline
- Still need Neo4j/Joern if scaling required

## Estimated Time Savings

| Phase | Original Estimate | With BRS | Savings |
|-------|-------------------|----------|---------|
| Phase 4 | 8 weeks | 2 weeks | **6 weeks** |
| Phase 3 | 4 weeks | 2 weeks | 2 weeks |
| Phase 5 | 6 weeks | 5 weeks | 1 week |
| Phase 7 | 4 weeks | 3 weeks | 1 week |
| **Total** | **22 weeks** | **12 weeks** | **10 weeks** |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| BRS SQLite won't scale | High | Add Neo4j adapter later |
| Type mappings imprecise | Medium | Extend BRS types as needed |
| BRS updates break integration | Medium | Pin version, fork if necessary |
| Missing explicit contraction | Low | Add method to brs.revision |

## Immediate Next Steps

1. **Add py-brs to curate-ipsum dependencies**
2. **Create `code_mutation` domain extension**
3. **Build mutation result → Evidence adapter**
4. **Test shadow evaluation with code patches**
5. **Prototype CEGIS loop using BRS infrastructure**

## Conclusion

BRS provides a solid foundation for curate-ipsum's belief revision needs. The key insight is that BRS is **domain-agnostic infrastructure** - we add the code synthesis domain on top. This separation of concerns (belief management vs. synthesis algorithms) is architecturally clean and reduces development time by approximately **10 weeks**.

The main gaps are:
1. Synthesis algorithms (CEGIS/CEGAR/genetic) - expected, not BRS's concern
2. Verification backends (Z3/KLEE) - orthogonal to belief revision
3. Explicit AGM contraction - small addition to BRS

The integration is straightforward: use BRS for what it does well (belief storage, revision, evaluation), add new modules for what it doesn't (synthesis, verification).
