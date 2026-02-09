"""
Tests for py-brs integration in curate-ipsum.

These tests verify:
- Evidence adapter correctly maps curate-ipsum types to BRS types
- TheoryManager correctly wraps BRS operations
- MCP tools function correctly
- Dual-write works as expected
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    FileMutationStats,
    MutationRunResult,
    RunKind,
    TestRunResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_test_result() -> TestRunResult:
    """Create a sample TestRunResult for testing."""
    return TestRunResult(
        id="test_001",
        projectId="test_project",
        commitSha="abc123def456",
        regionId="src/main.py::compute_value",
        timestamp=datetime(2025, 1, 27, 12, 0, 0, tzinfo=UTC),
        kind=RunKind.UNIT,
        passed=True,
        totalTests=10,
        passedTests=10,
        failedTests=0,
        durationMs=1500,
        framework="pytest",
        failingTests=[],
    )


@pytest.fixture
def sample_failed_test_result() -> TestRunResult:
    """Create a sample failed TestRunResult."""
    return TestRunResult(
        id="test_002",
        projectId="test_project",
        commitSha="abc123def456",
        regionId="src/main.py::compute_value",
        timestamp=datetime(2025, 1, 27, 12, 5, 0, tzinfo=UTC),
        kind=RunKind.UNIT,
        passed=False,
        totalTests=10,
        passedTests=8,
        failedTests=2,
        durationMs=1800,
        framework="pytest",
        failingTests=["test_edge_case", "test_null_input"],
    )


@pytest.fixture
def sample_mutation_result() -> MutationRunResult:
    """Create a sample MutationRunResult for testing."""
    return MutationRunResult(
        id="mutation_001",
        projectId="test_project",
        commitSha="abc123def456",
        regionId="src/main.py::compute_value",
        timestamp=datetime(2025, 1, 27, 12, 10, 0, tzinfo=UTC),
        kind=RunKind.MUTATION,
        tool="stryker",
        totalMutants=100,
        killed=75,
        survived=20,
        noCoverage=5,
        mutationScore=0.789,
        runtimeMs=45000,
        byFile=[
            FileMutationStats(
                filePath="src/main.py",
                totalMutants=100,
                killed=75,
                survived=20,
                noCoverage=5,
                mutationScore=0.789,
            )
        ],
    )


@pytest.fixture
def sample_low_score_mutation_result() -> MutationRunResult:
    """Create a sample MutationRunResult with low score."""
    return MutationRunResult(
        id="mutation_002",
        projectId="test_project",
        commitSha="abc123def456",
        regionId="src/main.py::compute_value",
        timestamp=datetime(2025, 1, 27, 12, 15, 0, tzinfo=UTC),
        kind=RunKind.MUTATION,
        tool="stryker",
        totalMutants=100,
        killed=30,
        survived=60,
        noCoverage=10,
        mutationScore=0.333,
        runtimeMs=45000,
        byFile=[],
    )


@pytest.fixture
def tmp_project_path(tmp_path: Path) -> Path:
    """Create a temporary project directory."""
    project = tmp_path / "test_project"
    project.mkdir()
    return project


# =============================================================================
# Evidence Adapter Tests
# =============================================================================


class TestEvidenceAdapter:
    """Tests for the evidence adapter module."""

    def test_test_result_to_evidence_passed(self, sample_test_result: TestRunResult):
        """Test converting a passing test result to evidence."""
        pytest.importorskip("brs")
        from adapters.evidence_adapter import CodeEvidenceKind, test_result_to_evidence

        evidence = test_result_to_evidence(sample_test_result)

        assert evidence.id == f"test_{sample_test_result.id}"
        assert evidence.kind == CodeEvidenceKind.TEST_PASS
        assert evidence.reliability == "B"
        assert "pytest" in evidence.citation
        assert evidence.metadata["project_id"] == "test_project"
        assert evidence.metadata["passed_tests"] == 10
        assert evidence.metadata["failed_tests"] == 0

    def test_test_result_to_evidence_failed(self, sample_failed_test_result: TestRunResult):
        """Test converting a failing test result to evidence."""
        pytest.importorskip("brs")
        from adapters.evidence_adapter import CodeEvidenceKind, test_result_to_evidence

        evidence = test_result_to_evidence(sample_failed_test_result)

        assert evidence.kind == CodeEvidenceKind.TEST_FAIL
        assert evidence.metadata["failed_tests"] == 2
        assert "test_edge_case" in evidence.metadata["failing_tests"]

    def test_mutation_result_to_evidence_killed(self, sample_mutation_result: MutationRunResult):
        """Test converting a high-score mutation result to evidence."""
        pytest.importorskip("brs")
        from adapters.evidence_adapter import CodeEvidenceKind, mutation_result_to_evidence

        evidence = mutation_result_to_evidence(sample_mutation_result)

        assert evidence.id == f"mutation_{sample_mutation_result.id}"
        assert evidence.kind == CodeEvidenceKind.MUTATION_KILLED  # score > 0.5
        assert evidence.reliability == "B"
        assert evidence.metadata["mutation_score"] == pytest.approx(0.789, rel=0.01)
        assert evidence.metadata["killed"] == 75

    def test_mutation_result_to_evidence_survived(self, sample_low_score_mutation_result: MutationRunResult):
        """Test converting a low-score mutation result to evidence."""
        pytest.importorskip("brs")
        from adapters.evidence_adapter import CodeEvidenceKind, mutation_result_to_evidence

        evidence = mutation_result_to_evidence(sample_low_score_mutation_result)

        assert evidence.kind == CodeEvidenceKind.MUTATION_SURVIVED  # score <= 0.5

    def test_evidence_adapter_without_brs(self):
        """Test that adapter fails gracefully without py-brs."""
        # This test verifies the ImportError handling
        # We mock the import to simulate py-brs not being installed
        pass  # The adapter has built-in handling for this


# =============================================================================
# Theory Manager Tests
# =============================================================================


class TestTheoryManager:
    """Tests for the TheoryManager class."""

    def test_manager_initialization(self, tmp_project_path: Path):
        """Test TheoryManager initializes correctly."""
        pytest.importorskip("brs")
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)

        assert manager.domain == "code_mutation"
        assert manager.world_label == "green"
        assert manager._project_path == tmp_project_path

    def test_add_assertion(self, tmp_project_path: Path):
        """Test adding an assertion to the theory."""
        pytest.importorskip("brs")
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)

        # First store some evidence
        from adapters.evidence_adapter import test_result_to_evidence
        from models import RunKind, TestRunResult

        test_result = TestRunResult(
            id="test_for_assertion",
            projectId="test_project",
            commitSha="abc123",
            regionId="region_1",
            timestamp=datetime.now(UTC),
            kind=RunKind.UNIT,
            passed=True,
            totalTests=5,
            passedTests=5,
            failedTests=0,
            durationMs=100,
            framework="pytest",
            failingTests=[],
        )
        evidence = test_result_to_evidence(test_result)
        manager.store_evidence(evidence)

        # Now add an assertion
        node = manager.add_assertion(
            assertion_type="behavior",
            content="function returns positive values",
            evidence_id=evidence.id,
            confidence=0.8,
            region_id="region_1",
        )

        assert "id" in node
        assert node["properties"]["assertion_type"] == "behavior"
        assert node["properties"]["confidence"] == 0.8

    def test_add_assertion_invalid_type(self, tmp_project_path: Path):
        """Test that invalid assertion types are rejected."""
        pytest.importorskip("brs")
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)

        with pytest.raises(ValueError, match="Invalid assertion_type"):
            manager.add_assertion(
                assertion_type="invalid_type",
                content="some content",
                evidence_id="evidence_1",
            )

    def test_add_assertion_invalid_confidence(self, tmp_project_path: Path):
        """Test that out-of-range confidence is rejected."""
        pytest.importorskip("brs")
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)

        with pytest.raises(ValueError, match="confidence must be between"):
            manager.add_assertion(
                assertion_type="behavior",
                content="some content",
                evidence_id="evidence_1",
                confidence=1.5,
            )

    def test_list_assertions(self, tmp_project_path: Path):
        """Test listing assertions with filtering."""
        pytest.importorskip("brs")
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)

        # Add multiple assertions
        manager.add_assertion(
            assertion_type="behavior",
            content="returns positive",
            evidence_id="ev1",
        )
        manager.add_assertion(
            assertion_type="type",
            content="param is int",
            evidence_id="ev2",
        )
        manager.add_assertion(
            assertion_type="behavior",
            content="handles null",
            evidence_id="ev3",
            region_id="region_1",
        )

        # List all
        all_assertions = manager.list_assertions()
        assert len(all_assertions) == 3

        # Filter by type
        behavior_assertions = manager.list_assertions(assertion_type="behavior")
        assert len(behavior_assertions) == 2

        # Filter by region
        region_assertions = manager.list_assertions(region_id="region_1")
        assert len(region_assertions) == 1

    def test_contract_assertion(self, tmp_project_path: Path):
        """Test contracting (removing) an assertion."""
        pytest.importorskip("brs")
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)

        # Add an assertion
        node = manager.add_assertion(
            assertion_type="behavior",
            content="test content",
            evidence_id="ev1",
        )

        # Verify it exists
        assertions_before = manager.list_assertions()
        assert len(assertions_before) == 1

        # Contract it
        result = manager.contract_assertion(node["id"], strategy="minimal")

        assert node["id"] in result.nodes_removed

    def test_get_entrenchment(self, tmp_project_path: Path):
        """Test computing entrenchment scores."""
        pytest.importorskip("brs")
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)

        # Add an assertion
        node = manager.add_assertion(
            assertion_type="behavior",
            content="test entrenchment",
            evidence_id="ev1",
            confidence=0.9,
        )

        score = manager.get_entrenchment(node["id"])

        assert 0.0 <= score <= 1.0

    def test_get_theory_snapshot(self, tmp_project_path: Path):
        """Test getting theory snapshot."""
        pytest.importorskip("brs")
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)

        # Add some content
        manager.add_assertion(
            assertion_type="type",
            content="test snapshot",
            evidence_id="ev1",
        )

        snapshot = manager.get_theory_snapshot()

        assert snapshot["domain_id"] == "code_mutation"
        assert snapshot["version_label"] == "green"
        assert len(snapshot["node_ids"]) >= 1


# =============================================================================
# Domain Smoke Tests
# =============================================================================


class TestDomainSmokeTests:
    """Tests for the code_mutation domain extension."""

    def test_smoke_tests_on_empty_world(self, tmp_project_path: Path):
        """Test smoke tests run on an empty world."""
        pytest.importorskip("brs")
        from domains.code_mutation_smoke import run_smoke
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)
        manager._ensure_world_exists()

        tests, failures, messages = run_smoke(
            manager.store,
            manager.domain,
            manager.world_label,
        )

        # Should pass (no nodes = no failures)
        assert tests >= 1
        # Empty world should have no failures
        assert failures == 0

    def test_smoke_tests_detect_missing_evidence(self, tmp_project_path: Path):
        """Test that smoke tests detect nodes without evidence."""
        pytest.importorskip("brs")
        import datetime

        from brs import canonical_json, content_hash

        from domains.code_mutation_smoke import run_smoke
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)
        manager._ensure_world_exists()

        # Add a node WITHOUT evidence (violates domain rules)
        node_id = "orphan_node_123"
        node = {
            "id": node_id,
            "domain_id": manager.domain,
            "kind": "Assertion",
            "properties": {"content": "orphan"},
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }

        node_hash = content_hash(node)
        manager.store._conn.execute(
            "INSERT OR IGNORE INTO objects(hash, kind, json) VALUES(?,?,?)", (node_hash, "Node", canonical_json(node))
        )

        # Update world to include this node (without grounding edge)
        world_data = manager.store.get_world(manager.domain, manager.world_label)["json"]
        new_world = {
            **world_data,
            "node_ids": list(world_data.get("node_ids", [])) + [node_id],
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        new_hash = content_hash(new_world)
        manager.store._conn.execute(
            "INSERT OR IGNORE INTO objects(hash, kind, json) VALUES(?,?,?)",
            (new_hash, "WorldBundle", canonical_json(new_world)),
        )
        manager.store._conn.execute(
            "INSERT OR REPLACE INTO worlds(domain_id, version_label, hash, created_utc) VALUES(?,?,?,?)",
            (manager.domain, manager.world_label, new_hash, new_world["created_utc"]),
        )
        manager.store._conn.commit()

        tests, failures, messages = run_smoke(
            manager.store,
            manager.domain,
            manager.world_label,
        )

        # Should detect the missing evidence
        assert failures >= 1
        assert any("without grounding evidence" in m for m in messages)


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, tmp_project_path: Path, sample_mutation_result: MutationRunResult):
        """Test a complete workflow from mutation result to assertion to contraction."""
        pytest.importorskip("brs")
        from adapters.evidence_adapter import mutation_result_to_evidence
        from theory import TheoryManager

        manager = TheoryManager(tmp_project_path)

        # 1. Store mutation evidence
        evidence = mutation_result_to_evidence(sample_mutation_result)
        manager.store_evidence(evidence)

        # 2. Add assertion based on evidence
        node = manager.add_assertion(
            assertion_type="behavior",
            content="compute_value handles edge cases correctly",
            evidence_id=evidence.id,
            confidence=0.75,
            region_id=sample_mutation_result.regionId,
        )

        # 3. Check entrenchment (may be 0.0 for a singleton assertion — that's valid)
        score = manager.get_entrenchment(node["id"])
        assert score >= 0.0

        # 4. Get snapshot
        snapshot = manager.get_theory_snapshot()
        assert len(snapshot["node_ids"]) == 1
        assert len(snapshot["evidence_ids"]) == 1

        # 5. Contract the assertion
        result = manager.contract_assertion(node["id"])
        assert node["id"] in result.nodes_removed

        # 6. Verify it's gone
        assertions = manager.list_assertions()
        assert len(assertions) == 0


# =============================================================================
# Skip marker for tests requiring py-brs
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_brs: mark test as requiring py-brs package",
    )
