"""
Persistent storage for synthesis results.

Uses JSONL (newline-delimited JSON) for append-only persistence,
mirroring the project's existing tools.py::append_run() pattern.

Each line is a JSON object with the SynthesisResult fields plus
a project_id key for multi-project filtering.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from synthesis.models import (
    CodePatch,
    PatchSource,
    SynthesisResult,
    SynthesisStatus,
)

LOG = logging.getLogger("storage.synthesis_store")


class SynthesisStore:
    """Append-only JSONL store for synthesis results."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._file = data_dir / "synthesis_runs.jsonl"

    def _ensure_dir(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def append(self, result: SynthesisResult, project_id: str) -> None:
        """Append a synthesis result to the JSONL store."""
        self._ensure_dir()
        payload = result.to_dict()
        payload["project_id"] = project_id
        payload["stored_at"] = datetime.now(timezone.utc).isoformat()

        with self._file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        LOG.debug("Stored synthesis result %s for project %s", result.id, project_id)

    def load_all(self, project_id: str) -> list[SynthesisResult]:
        """Load all synthesis results for a project."""
        results = []
        for record in self._iter_records():
            if record.get("project_id") == project_id:
                results.append(self._record_to_result(record))
        return results

    def load_by_id(self, synthesis_id: str) -> SynthesisResult | None:
        """Load a specific synthesis result by ID."""
        for record in self._iter_records():
            if record.get("id") == synthesis_id:
                return self._record_to_result(record)
        return None

    def load_by_region(self, project_id: str, region_id: str) -> list[SynthesisResult]:
        """Load all synthesis results for a specific region within a project."""
        results = []
        for record in self._iter_records():
            if record.get("project_id") != project_id:
                continue
            patch = record.get("patch")
            if patch and patch.get("region_id") == region_id:
                results.append(self._record_to_result(record))
        return results

    def _iter_records(self):
        """Iterate over all raw JSON records in the store."""
        if not self._file.exists():
            return

        with self._file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    LOG.warning("Skipping malformed line %d in %s", line_num, self._file)

    @staticmethod
    def _record_to_result(record: dict) -> SynthesisResult:
        """Convert a raw JSON record back to a SynthesisResult."""
        patch_data = record.get("patch")
        patch = None
        if patch_data:
            patch = CodePatch(
                code=patch_data.get("code", ""),
                source=PatchSource(patch_data.get("source", "llm")),
                diff=patch_data.get("diff", ""),
                region_id=patch_data.get("region_id", ""),
                metadata=patch_data.get("metadata", {}),
            )

        return SynthesisResult(
            id=record.get("id", ""),
            status=SynthesisStatus(record.get("status", "failed")),
            patch=patch,
            iterations=record.get("iterations", 0),
            counterexamples_resolved=record.get("counterexamples_resolved", 0),
            duration_ms=record.get("duration_ms", 0),
            fitness_history=record.get("fitness_history", []),
            final_entropy=record.get("final_entropy", 0.0),
            total_candidates_evaluated=record.get("total_candidates_evaluated", 0),
            error_message=record.get("error_message", ""),
        )
