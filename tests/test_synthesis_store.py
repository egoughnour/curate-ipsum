"""Tests for storage.synthesis_store — JSONL persistence for synthesis results."""

import pytest

from storage.synthesis_store import SynthesisStore
from synthesis.models import (
    CodePatch,
    PatchSource,
    SynthesisResult,
    SynthesisStatus,
)


@pytest.fixture
def store(tmp_path):
    """Create a SynthesisStore using a temp directory."""
    return SynthesisStore(tmp_path)


def _make_result(
    status=SynthesisStatus.SUCCESS,
    result_id="test-01",
    region_id="region_a",
    iterations=5,
) -> SynthesisResult:
    return SynthesisResult(
        id=result_id,
        status=status,
        patch=CodePatch(
            code="def patched(): return 42",
            source=PatchSource.LLM,
            region_id=region_id,
        ),
        iterations=iterations,
        counterexamples_resolved=3,
        duration_ms=1500,
        fitness_history=[0.1, 0.4, 0.7, 0.9, 0.95],
        final_entropy=1.8,
        total_candidates_evaluated=40,
    )


class TestSynthesisStore:
    """Tests for SynthesisStore."""

    def test_append_and_load_roundtrip(self, store):
        """Append a result and load it back."""
        result = _make_result()
        store.append(result, project_id="proj-1")

        loaded = store.load_all("proj-1")
        assert len(loaded) == 1
        assert loaded[0].id == "test-01"
        assert loaded[0].status == SynthesisStatus.SUCCESS
        assert loaded[0].iterations == 5
        assert loaded[0].patch is not None
        assert loaded[0].patch.code == "def patched(): return 42"

    def test_load_by_id(self, store):
        """Load a specific result by synthesis ID."""
        r1 = _make_result(result_id="aaa")
        r2 = _make_result(result_id="bbb", status=SynthesisStatus.FAILED)
        store.append(r1, "proj-1")
        store.append(r2, "proj-1")

        loaded = store.load_by_id("bbb")
        assert loaded is not None
        assert loaded.id == "bbb"
        assert loaded.status == SynthesisStatus.FAILED

    def test_load_by_id_not_found(self, store):
        """Loading a non-existent ID returns None."""
        assert store.load_by_id("missing") is None

    def test_load_by_region(self, store):
        """Load results filtered by region ID."""
        r1 = _make_result(result_id="r1", region_id="region_a")
        r2 = _make_result(result_id="r2", region_id="region_b")
        r3 = _make_result(result_id="r3", region_id="region_a")
        store.append(r1, "proj-1")
        store.append(r2, "proj-1")
        store.append(r3, "proj-1")

        by_region = store.load_by_region("proj-1", "region_a")
        assert len(by_region) == 2
        assert {r.id for r in by_region} == {"r1", "r3"}

    def test_empty_store_returns_empty(self, store):
        """Empty store returns empty list."""
        assert store.load_all("any-project") == []

    def test_multiple_projects_dont_cross(self, store):
        """Results from different projects don't cross-contaminate."""
        r1 = _make_result(result_id="p1-result")
        r2 = _make_result(result_id="p2-result")
        store.append(r1, "project-alpha")
        store.append(r2, "project-beta")

        alpha = store.load_all("project-alpha")
        beta = store.load_all("project-beta")

        assert len(alpha) == 1
        assert alpha[0].id == "p1-result"
        assert len(beta) == 1
        assert beta[0].id == "p2-result"

    def test_fitness_history_preserved(self, store):
        """Fitness history list survives round-trip."""
        result = _make_result()
        store.append(result, "proj-1")

        loaded = store.load_all("proj-1")[0]
        assert loaded.fitness_history == [0.1, 0.4, 0.7, 0.9, 0.95]
        assert loaded.final_entropy == 1.8
        assert loaded.total_candidates_evaluated == 40

    def test_result_without_patch(self, store):
        """A failed result with no patch serializes correctly."""
        result = SynthesisResult(
            id="fail-01",
            status=SynthesisStatus.FAILED,
            error_message="No valid candidates",
        )
        store.append(result, "proj-1")

        loaded = store.load_all("proj-1")[0]
        assert loaded.id == "fail-01"
        assert loaded.patch is None
        assert loaded.error_message == "No valid candidates"

    def test_data_dir_created_automatically(self, tmp_path):
        """Store creates the data directory if it doesn't exist."""
        sub = tmp_path / "nested" / "deep"
        store = SynthesisStore(sub)
        store.append(_make_result(), "proj-1")

        assert sub.exists()
        assert (sub / "synthesis_runs.jsonl").exists()

    def test_malformed_line_skipped(self, tmp_path):
        """Malformed JSONL lines are skipped without error."""
        store = SynthesisStore(tmp_path)
        # Write a valid result
        store.append(_make_result(result_id="good"), "proj-1")

        # Inject a malformed line
        jsonl_path = tmp_path / "synthesis_runs.jsonl"
        with jsonl_path.open("a") as f:
            f.write("this is not json\n")

        # Another valid result
        store.append(_make_result(result_id="also-good"), "proj-1")

        loaded = store.load_all("proj-1")
        assert len(loaded) == 2
        assert {r.id for r in loaded} == {"good", "also-good"}
