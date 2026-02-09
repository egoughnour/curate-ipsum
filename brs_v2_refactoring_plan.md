# py-brs v2.0.0 Integration: Refactoring Plan

## Executive Summary

**No obstacles exist** to taking `py-brs>=2.0.0` as a dependency. The v2.0.0 release closes all identified gaps (AGM contraction, revision, entrenchment scoring). This plan outlines the refactoring needed to integrate py-brs into curate-ipsum.

## Phase 0: Add Dependency (Day 1)

### Task 0.1: Create pyproject.toml

curate-ipsum currently lacks a `pyproject.toml`. Create one:

```toml
[project]
name = "curate-ipsum"
version = "0.1.0"
description = "Mutation testing orchestration MCP server with belief revision"
requires-python = ">=3.10"
dependencies = [
    "py-brs>=2.0.0",
    "pydantic>=2.0",
    "mcp[fastmcp]>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "ruff>=0.1.0",
]

[project.scripts]
curate-ipsum = "server:main"
```

### Task 0.2: Verify Import

```python
# Quick verification
from brs import CASStore, Node, Edge, Evidence, WorldBundle
from brs.revision import contract, revise, compute_entrenchment
```

---

## Phase 1: Adapter Layer (Days 2-3)

### Task 1.1: Create `adapters/` Module

```
curate_ipsum/
├── adapters/
│   ├── __init__.py
│   ├── evidence_adapter.py    # RunResult → Evidence
│   ├── theory_adapter.py      # SynthesisTheory ↔ WorldBundle
│   └── metrics_adapter.py     # RegionMetrics → Maturity
├── models.py                  # Existing
├── tools.py                   # Existing
└── server.py                  # Existing
```

### Task 1.2: Evidence Adapter

```python
# adapters/evidence_adapter.py

from brs import Evidence
from models import TestRunResult, MutationRunResult

# Evidence kind mapping
class CodeEvidenceKind:
    TEST_PASS = "test_pass"
    TEST_FAIL = "test_fail"
    MUTATION_KILLED = "mutation_killed"
    MUTATION_SURVIVED = "mutation_survived"

# Reliability mapping
RELIABILITY_MAP = {
    "test_pass": "B",
    "test_fail": "B",
    "mutation_killed": "B",
    "mutation_survived": "B",
}

def test_result_to_evidence(run: TestRunResult) -> Evidence:
    """Convert TestRunResult to BRS Evidence."""
    kind = CodeEvidenceKind.TEST_PASS if run.passed else CodeEvidenceKind.TEST_FAIL
    return Evidence(
        id=f"test_{run.id}",
        citation=f"{run.framework} test run at {run.timestamp.isoformat()}",
        kind=kind,
        reliability=RELIABILITY_MAP[kind],
        date=run.timestamp.isoformat(),
        metadata={
            "project_id": run.projectId,
            "commit_sha": run.commitSha,
            "region_id": run.regionId,
            "total_tests": run.totalTests,
            "passed_tests": run.passedTests,
            "failed_tests": run.failedTests,
            "failing_tests": run.failingTests,
            "duration_ms": run.durationMs,
        }
    )

def mutation_result_to_evidence(run: MutationRunResult) -> Evidence:
    """Convert MutationRunResult to BRS Evidence."""
    # Determine primary kind based on mutation score
    kind = (
        CodeEvidenceKind.MUTATION_KILLED
        if run.mutationScore > 0.5
        else CodeEvidenceKind.MUTATION_SURVIVED
    )
    return Evidence(
        id=f"mutation_{run.id}",
        citation=f"{run.tool} mutation run at {run.timestamp.isoformat()}",
        kind=kind,
        reliability=RELIABILITY_MAP[kind],
        date=run.timestamp.isoformat(),
        metadata={
            "project_id": run.projectId,
            "commit_sha": run.commitSha,
            "region_id": run.regionId,
            "tool": run.tool,
            "total_mutants": run.totalMutants,
            "killed": run.killed,
            "survived": run.survived,
            "no_coverage": run.noCoverage,
            "mutation_score": run.mutationScore,
            "runtime_ms": run.runtimeMs,
            "by_file": [f.model_dump() for f in run.byFile],
        }
    )
```

### Task 1.3: Storage Migration Helper

```python
# adapters/storage_migration.py

from pathlib import Path
from brs import CASStore
from tools import _load_runs
from .evidence_adapter import test_result_to_evidence, mutation_result_to_evidence

def migrate_jsonl_to_casstore(
    jsonl_dir: Path,
    casstore_path: Path,
    domain: str = "code_mutation"
) -> int:
    """
    Migrate existing JSONL run history to CASStore.

    Returns number of records migrated.
    """
    runs = _load_runs()
    store = CASStore(casstore_path)

    migrated = 0
    for run in runs:
        if hasattr(run, 'tool'):  # MutationRunResult
            evidence = mutation_result_to_evidence(run)
        else:  # TestRunResult
            evidence = test_result_to_evidence(run)

        # Store evidence in CASStore
        store.put_evidence(domain, evidence)
        migrated += 1

    return migrated
```

---

## Phase 2: Code Mutation Domain (Days 4-5)

### Task 2.1: Domain Extension

Create a BRS domain extension for code mutation testing:

```python
# domains/code_mutation_smoke.py
"""
BRS domain extension for code mutation testing.
Register with: brs.domains.registry
"""

DOMAIN_ID = "code_mutation"

def run_smoke(store, domain_id, world_label):
    """Basic smoke tests for code mutation domain."""
    tests, failures, messages = 0, 0, []

    # Test 1: All assertions have evidence
    tests += 1
    nodes = store.list_nodes(domain=domain_id)
    for node in nodes:
        edges = store.get_edges_to(node.id)
        evidence_edges = [e for e in edges if e.kind == "grounded_by"]
        if not evidence_edges:
            failures += 1
            messages.append(f"Node {node.id} has no grounding evidence")

    # Test 2: Mutation score monotonicity (optional check)
    tests += 1
    # Check that mutation scores generally improve over time
    # (implementation depends on how we track score history)

    # Test 3: No contradictory assertions
    tests += 1
    # Check for nodes that contradict each other
    # (e.g., "function returns int" vs "function returns str")

    return (tests, failures, messages)

def run_regression(store, domain_id, world_label):
    """Regression tests ensuring no previously-killed mutants resurface."""
    tests, failures, messages = 0, 0, []

    # Get all killed mutants from history
    # Verify none have "survived" evidence after "killed" evidence

    return (tests, failures, messages)

SMOKE_LEVELS = {
    "smoke": run_smoke,
    "regression": run_regression,
}
```

### Task 2.2: Register Domain

```python
# In curate_ipsum initialization
from brs.domains.registry import register_domain_smoke

# Register our domain extension
register_domain_smoke(
    "code_mutation",
    "domains.code_mutation_smoke",
    "run_smoke"
)
```

---

## Phase 3: Belief Revision Integration (Days 6-8)

### Task 3.1: Theory Manager

```python
# theory/manager.py

from pathlib import Path
from typing import Optional, List
from brs import CASStore, Node, Edge, WorldBundle
from brs.revision import contract, revise, compute_entrenchment, ContractionResult

class TheoryManager:
    """
    Manages belief revision operations for code synthesis.
    Wraps py-brs CASStore with curate-ipsum-specific logic.
    """

    def __init__(self, project_path: Path, domain: str = "code_mutation"):
        self.store = CASStore(project_path / ".curate_ipsum" / "beliefs.db")
        self.domain = domain
        self.world_label = "green"  # Start with "green" world

    def add_assertion(
        self,
        assertion_type: str,
        content: str,
        evidence_id: str,
        confidence: float = 0.5
    ) -> Node:
        """Add a new assertion (belief) to the theory."""
        node = Node(
            id=f"{assertion_type}_{hash(content)}",
            domain=self.domain,
            properties={
                "type": assertion_type,  # "type", "behavior", "invariant"
                "content": content,
                "confidence": confidence,
            }
        )

        # Create edge to evidence
        edge = Edge(
            source_id=node.id,
            target_id=evidence_id,
            kind="grounded_by",
            tier=3,  # Adjust based on evidence reliability
            confidence=confidence,
        )

        self.store.put_node(self.domain, node)
        self.store.put_edge(self.domain, edge)

        return node

    def contract_assertion(
        self,
        node_id: str,
        strategy: str = "entrenchment"
    ) -> ContractionResult:
        """Remove an assertion using AGM contraction."""
        return contract(
            self.store,
            node_id,
            strategy=strategy,
            domain=self.domain
        )

    def revise_with_evidence(
        self,
        new_belief: Node,
        contradicted_belief_id: str,
        evidence: "Evidence"
    ) -> None:
        """Revise theory: contract contradicted belief, add new belief."""
        # First contract the old belief
        self.contract_assertion(contradicted_belief_id, strategy="entrenchment")

        # Then add the new belief
        self.store.put_node(self.domain, new_belief)
        self.store.put_evidence(self.domain, evidence)

    def get_entrenchment(self, node_id: str) -> float:
        """Get entrenchment score for a belief."""
        return compute_entrenchment(self.store, node_id)

    def get_theory_snapshot(self) -> WorldBundle:
        """Get current theory state as immutable snapshot."""
        return self.store.get_world(self.domain, self.world_label)

    def list_assertions(
        self,
        assertion_type: Optional[str] = None
    ) -> List[Node]:
        """List all assertions, optionally filtered by type."""
        nodes = self.store.list_nodes(domain=self.domain)
        if assertion_type:
            nodes = [n for n in nodes if n.properties.get("type") == assertion_type]
        return nodes
```

### Task 3.2: MCP Tool Integration

Add new MCP tools that expose belief revision capabilities:

```python
# server.py additions

from theory.manager import TheoryManager

# ... existing code ...

def build_server() -> "FastMCP":
    server = _require_server()

    # ... existing tools ...

    @server.tool(
        description="Add an assertion to the synthesis theory with evidence grounding."
    )
    async def add_assertion_tool(
        projectId: str,
        assertionType: str,  # "type", "behavior", "invariant"
        content: str,
        evidenceId: str,
        confidence: float = 0.5,
    ) -> dict:
        _validate_required("projectId", projectId)
        _validate_required("assertionType", assertionType)
        _validate_required("content", content)
        _validate_required("evidenceId", evidenceId)

        manager = TheoryManager(Path(DATA_DIR) / projectId)
        node = manager.add_assertion(
            assertion_type=assertionType,
            content=content,
            evidence_id=evidenceId,
            confidence=confidence,
        )
        return {"node_id": node.id, "status": "added"}

    @server.tool(
        description="Contract (remove) an assertion from the theory using AGM contraction."
    )
    async def contract_assertion_tool(
        projectId: str,
        nodeId: str,
        strategy: str = "entrenchment",  # "entrenchment", "minimal", "full_cascade"
    ) -> dict:
        _validate_required("projectId", projectId)
        _validate_required("nodeId", nodeId)

        manager = TheoryManager(Path(DATA_DIR) / projectId)
        result = manager.contract_assertion(nodeId, strategy=strategy)
        return {
            "removed_nodes": result.removed_nodes,
            "removed_edges": result.removed_edges,
            "strategy": strategy,
        }

    @server.tool(
        description="Get entrenchment score for an assertion (0.0-1.0, higher = more entrenched)."
    )
    async def get_entrenchment_tool(
        projectId: str,
        nodeId: str,
    ) -> dict:
        _validate_required("projectId", projectId)
        _validate_required("nodeId", nodeId)

        manager = TheoryManager(Path(DATA_DIR) / projectId)
        score = manager.get_entrenchment(nodeId)
        return {"node_id": nodeId, "entrenchment": score}

    @server.tool(
        description="List all assertions in the theory, optionally filtered by type."
    )
    async def list_assertions_tool(
        projectId: str,
        assertionType: Optional[str] = None,
    ) -> dict:
        _validate_required("projectId", projectId)

        manager = TheoryManager(Path(DATA_DIR) / projectId)
        nodes = manager.list_assertions(assertion_type=assertionType)
        return {
            "assertions": [
                {
                    "id": n.id,
                    "type": n.properties.get("type"),
                    "content": n.properties.get("content"),
                    "confidence": n.properties.get("confidence"),
                }
                for n in nodes
            ]
        }

    return server
```

---

## Phase 4: Refactor Existing Tools (Days 9-10)

### Task 4.1: Dual-Write Strategy

During transition, write to both JSONL (backward compat) and CASStore:

```python
# tools.py modifications

from adapters.evidence_adapter import test_result_to_evidence, mutation_result_to_evidence
from brs import CASStore

# Add CASStore alongside existing JSONL
CASSTORE_DIR = DATA_DIR / "beliefs"

def append_run(run: RunResult) -> None:
    """Append run to both JSONL (legacy) and CASStore."""
    _ensure_data_dir()

    # Legacy JSONL write
    payload = run.model_dump(mode="json")
    with RUNS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

    # New CASStore write
    store = CASStore(CASSTORE_DIR)
    if isinstance(run, MutationRunResult):
        evidence = mutation_result_to_evidence(run)
    else:
        evidence = test_result_to_evidence(run)
    store.put_evidence("code_mutation", evidence)
```

### Task 4.2: Migrate RegionMetrics to Use Entrenchment

```python
# tools.py - enhanced metrics

from brs.revision import compute_entrenchment

def compute_region_metrics(
    project_id: str,
    commit_sha: str,
    region_id: str,
    history: List[RunResult]
) -> RegionMetrics:
    """Compute region metrics with BRS entrenchment integration."""

    # ... existing PID calculation ...

    # NEW: Compute entrenchment-weighted metrics
    store = CASStore(CASSTORE_DIR)

    # Get all assertions related to this region
    region_nodes = [
        n for n in store.list_nodes(domain="code_mutation")
        if n.properties.get("region_id") == region_id
    ]

    # Compute average entrenchment as "stability" metric
    if region_nodes:
        entrenchments = [compute_entrenchment(store, n.id) for n in region_nodes]
        stability = sum(entrenchments) / len(entrenchments)
    else:
        stability = 0.0

    return RegionMetrics(
        projectId=project_id,
        commitSha=commit_sha,
        regionId=region_id,
        mutationScore=scores[-1],
        centrality=0.5,  # TODO: compute from graph
        triviality=0.5,  # TODO: compute from graph
        pid=pid,
        # NEW field:
        # stability=stability,
    )
```

---

## Phase 5: Documentation & Testing (Days 11-12)

### Task 5.1: Update README

Add section on belief revision capabilities:

```markdown
## Belief Revision

curate-ipsum uses [py-brs](https://pypi.org/project/py-brs/) for AGM-compliant belief revision:

### MCP Tools

- `add_assertion` - Add typed assertions (type/behavior/invariant) with evidence
- `contract_assertion` - Remove beliefs using entrenchment-guided contraction
- `get_entrenchment` - Query belief resilience (0.0-1.0 scale)
- `list_assertions` - View current theory state

### Evidence Types

| Evidence | Reliability | Description |
|----------|-------------|-------------|
| Test pass/fail | B | Dynamic test execution |
| Mutation killed/survived | B | Mutation testing outcome |
| Type check | B | Static type analysis |
| SMT proof | A | Formal verification |
| LLM suggestion | C | Statistical plausibility only |
```

### Task 5.2: Test Suite

```python
# tests/test_brs_integration.py

import pytest
from pathlib import Path
from theory.manager import TheoryManager
from adapters.evidence_adapter import test_result_to_evidence
from models import TestRunResult, RunKind
from datetime import datetime, timezone

@pytest.fixture
def manager(tmp_path):
    return TheoryManager(tmp_path)

def test_add_and_contract_assertion(manager):
    # Add an assertion
    node = manager.add_assertion(
        assertion_type="type",
        content="function returns int",
        evidence_id="test_123",
        confidence=0.8
    )

    assert node.id is not None

    # Contract it
    result = manager.contract_assertion(node.id, strategy="minimal")

    assert node.id in result.removed_nodes

def test_entrenchment_increases_with_evidence(manager):
    # Add assertion with weak evidence
    node1 = manager.add_assertion(
        assertion_type="behavior",
        content="handles null input",
        evidence_id="llm_1",
        confidence=0.3
    )

    score1 = manager.get_entrenchment(node1.id)

    # Add more evidence for same assertion
    # (In practice, this would be a separate evidence node)

    # Higher confidence evidence should increase entrenchment
    # ...

def test_evidence_adapter_roundtrip():
    run = TestRunResult(
        id="test_001",
        projectId="proj_1",
        commitSha="abc123",
        regionId="region_1",
        timestamp=datetime.now(timezone.utc),
        kind=RunKind.UNIT,
        passed=True,
        totalTests=10,
        passedTests=10,
        failedTests=0,
        durationMs=1500,
        framework="pytest",
        failingTests=[],
    )

    evidence = test_result_to_evidence(run)

    assert evidence.kind == "test_pass"
    assert evidence.reliability == "B"
    assert evidence.metadata["project_id"] == "proj_1"
```

---

## Timeline Summary

| Phase | Days | Deliverable |
|-------|------|-------------|
| 0: Dependency | 1 | pyproject.toml with py-brs>=2.0.0 |
| 1: Adapters | 2-3 | evidence_adapter.py, storage migration |
| 2: Domain | 4-5 | code_mutation domain extension |
| 3: Theory | 6-8 | TheoryManager + MCP tools |
| 4: Refactor | 9-10 | Dual-write, metrics enhancement |
| 5: Docs/Tests | 11-12 | README, test suite |

**Total: 12 days** (vs. original 8-week estimate for Phase 4 alone)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CASStore performance at scale | Start with SQLite, can migrate to Neo4j later |
| Type mapping imprecision | Extend BRS types as needed via properties dict |
| Breaking changes in py-brs | Pin version, vendor if necessary |
| Dual-write complexity | Clear migration path with rollback capability |

---

## Success Criteria

1. ✅ `pip install py-brs>=2.0.0` works
2. ✅ All existing tests pass
3. ✅ New MCP tools functional
4. ✅ Evidence properly stored in CASStore
5. ✅ Contraction/revision operations work correctly
6. ✅ Entrenchment scoring integrated with metrics
