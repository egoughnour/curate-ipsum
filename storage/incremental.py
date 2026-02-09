"""
Incremental update engine for graph persistence.

Detects which files changed since the last extraction and updates
only the affected graph nodes/edges, avoiding full re-extraction.

Uses SHA-256 file hashing to detect changes. The hash map is persisted
via the GraphStore's file_hashes table.

Decision: D-015
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from storage.graph_store import GraphStore

LOG = logging.getLogger("storage.incremental")


@dataclass
class ChangeSet:
    """Files that changed since last extraction."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.removed)

    @property
    def total_changed(self) -> int:
        return len(self.added) + len(self.modified) + len(self.removed)

    def to_dict(self) -> dict:
        return {
            "added": self.added,
            "modified": self.modified,
            "removed": self.removed,
            "total_changed": self.total_changed,
        }


@dataclass
class UpdateResult:
    """Result of an incremental graph update."""

    added_nodes: int = 0
    removed_nodes: int = 0
    modified_files: int = 0
    total_files_scanned: int = 0
    duration_ms: int = 0
    change_set: ChangeSet | None = None
    full_rebuild: bool = False

    def to_dict(self) -> dict:
        return {
            "added_nodes": self.added_nodes,
            "removed_nodes": self.removed_nodes,
            "modified_files": self.modified_files,
            "total_files_scanned": self.total_files_scanned,
            "duration_ms": self.duration_ms,
            "change_set": self.change_set.to_dict() if self.change_set else None,
            "full_rebuild": self.full_rebuild,
        }


class IncrementalEngine:
    """
    Detects file changes and performs incremental graph updates.

    Workflow:
    1. Compute current file hashes for all matching files
    2. Compare with stored hashes → ChangeSet
    3. For removed files: delete nodes/edges
    4. For added/modified files: re-extract and merge
    5. Update stored file hashes
    """

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    @staticmethod
    def compute_file_hashes(
        directory: Path,
        pattern: str = "**/*.py",
    ) -> dict[str, str]:
        """
        Compute SHA-256 hashes for all files matching pattern.

        Args:
            directory: Root directory to scan
            pattern: Glob pattern for files

        Returns:
            Dict mapping relative file paths to their SHA-256 hex digests
        """
        hashes: dict[str, str] = {}
        dir_path = Path(directory)

        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                try:
                    content = file_path.read_bytes()
                    digest = hashlib.sha256(content).hexdigest()
                    rel_path = str(file_path.relative_to(dir_path))
                    hashes[rel_path] = digest
                except (OSError, PermissionError) as exc:
                    LOG.warning("Cannot hash %s: %s", file_path, exc)

        return hashes

    def detect_changes(
        self,
        project_id: str,
        current_hashes: dict[str, str],
    ) -> ChangeSet:
        """
        Compare current file hashes with stored hashes to find changes.

        Args:
            project_id: Project identifier
            current_hashes: Current file → hash mapping

        Returns:
            ChangeSet with added, modified, and removed files
        """
        stored_hashes = self._store.get_file_hashes(project_id)

        current_files = set(current_hashes.keys())
        stored_files = set(stored_hashes.keys())

        added = sorted(current_files - stored_files)
        removed = sorted(stored_files - current_files)
        modified = sorted(f for f in current_files & stored_files if current_hashes[f] != stored_hashes[f])

        change_set = ChangeSet(added=added, modified=modified, removed=removed)

        LOG.info(
            "Change detection for %s: %d added, %d modified, %d removed",
            project_id,
            len(added),
            len(modified),
            len(removed),
        )

        return change_set

    def update_graph(
        self,
        project_id: str,
        directory: Path,
        pattern: str = "**/*.py",
        extractor_func=None,
    ) -> UpdateResult:
        """
        Perform an incremental graph update.

        Args:
            project_id: Project identifier
            directory: Root directory of the project
            pattern: File glob pattern
            extractor_func: Optional callable(file_path) → (nodes, edges) for extraction.
                If None, only file hash tracking and node deletion are performed.

        Returns:
            UpdateResult with counts of changes made
        """
        start = time.monotonic()
        result = UpdateResult()

        # Step 1: Compute current hashes
        current_hashes = self.compute_file_hashes(directory, pattern)
        result.total_files_scanned = len(current_hashes)

        # Step 2: Detect changes
        change_set = self.detect_changes(project_id, current_hashes)
        result.change_set = change_set

        if not change_set.has_changes:
            result.duration_ms = int((time.monotonic() - start) * 1000)
            LOG.info("No changes detected for project %s", project_id)
            return result

        # Step 3: Remove nodes for deleted files
        for file_path in change_set.removed:
            removed = self._store.delete_nodes_by_file(file_path, project_id)
            result.removed_nodes += removed

        # Step 4: Remove nodes for modified files (will be re-extracted)
        for file_path in change_set.modified:
            removed = self._store.delete_nodes_by_file(file_path, project_id)
            result.removed_nodes += removed

        # Step 5: Re-extract added and modified files
        files_to_extract = change_set.added + change_set.modified
        result.modified_files = len(files_to_extract)

        if extractor_func and files_to_extract:
            for file_path in files_to_extract:
                try:
                    full_path = Path(directory) / file_path
                    nodes, edges = extractor_func(str(full_path))

                    for node_data in nodes:
                        self._store.store_node(node_data, project_id)
                        result.added_nodes += 1

                    for edge_data in edges:
                        self._store.store_edge(edge_data, project_id)

                except Exception as exc:
                    LOG.warning("Failed to extract %s: %s", file_path, exc)

        # Step 6: Update stored file hashes
        # Remove hashes for deleted files
        updated_hashes = dict(current_hashes.items())
        self._store.set_file_hashes(project_id, updated_hashes)

        result.duration_ms = int((time.monotonic() - start) * 1000)
        LOG.info(
            "Incremental update for %s: +%d nodes, -%d nodes, %d files, %dms",
            project_id,
            result.added_nodes,
            result.removed_nodes,
            result.modified_files,
            result.duration_ms,
        )

        return result

    def force_full_rebuild(
        self,
        project_id: str,
        graph,
        directory: Path,
        pattern: str = "**/*.py",
    ) -> UpdateResult:
        """
        Force a complete graph rebuild (drop all + store full graph).

        Args:
            project_id: Project identifier
            graph: The complete CallGraph to persist
            directory: Root directory for file hash computation
            pattern: File glob pattern

        Returns:
            UpdateResult marked as full_rebuild
        """
        start = time.monotonic()

        # Store the entire graph (replaces existing)
        self._store.store_graph(graph, project_id)

        # Compute and store file hashes
        hashes = self.compute_file_hashes(directory, pattern)
        self._store.set_file_hashes(project_id, hashes)

        result = UpdateResult(
            added_nodes=len(graph.nodes),
            removed_nodes=0,
            modified_files=len(hashes),
            total_files_scanned=len(hashes),
            duration_ms=int((time.monotonic() - start) * 1000),
            full_rebuild=True,
        )

        LOG.info(
            "Full rebuild for %s: %d nodes, %d files, %dms",
            project_id,
            result.added_nodes,
            result.modified_files,
            result.duration_ms,
        )

        return result
