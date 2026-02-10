"""
Abstract graph store interface.

Defines the GraphStore ABC with two backends:
- SQLiteGraphStore (primary, zero dependencies)
- KuzuGraphStore (optional, embedded graph DB with Cypher)

Follows the D-012 pattern (hybrid client with abstract base + concrete backends).

Decision: D-014
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from curate_ipsum.graph.models import CallGraph

LOG = logging.getLogger("storage.graph_store")

# Storage directory name (alongside existing beliefs.db)
STORAGE_DIR_NAME = ".curate_ipsum"


class GraphStore(ABC):
    """
    Abstract interface for persistent graph storage.

    Implementations persist call graphs, reachability indices,
    Fiedler partitions, and file hashes for incremental updates.
    """

    @abstractmethod
    def store_graph(self, graph: CallGraph, project_id: str) -> None:
        """Persist an entire call graph (nodes + edges)."""

    @abstractmethod
    def load_graph(self, project_id: str) -> CallGraph | None:
        """Load a previously stored call graph."""

    @abstractmethod
    def store_node(self, node_data: dict[str, Any], project_id: str) -> None:
        """Store or update a single node."""

    @abstractmethod
    def store_edge(self, edge_data: dict[str, Any], project_id: str) -> None:
        """Store or update a single edge."""

    @abstractmethod
    def get_node(self, node_id: str, project_id: str) -> dict[str, Any] | None:
        """Get a single node's data by ID."""

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        project_id: str,
        direction: str = "outgoing",
        edge_kind: str | None = None,
    ) -> list[str]:
        """
        Get neighboring node IDs.

        Args:
            node_id: Source node
            project_id: Project identifier
            direction: "outgoing", "incoming", or "both"
            edge_kind: Filter by edge kind (None = all)
        """

    @abstractmethod
    def query_reachable(self, source_id: str, target_id: str, project_id: str) -> bool:
        """
        Check if target is reachable from source using stored Kameda labels.

        Falls back to non-planar reachability table if needed.
        """

    @abstractmethod
    def store_reachability_index(
        self,
        kameda_data: dict[str, Any],
        project_id: str,
    ) -> None:
        """
        Persist Kameda reachability index.

        kameda_data keys: left_rank, right_rank, source_id, sink_id,
                          non_planar_reachability, all_node_ids
        """

    @abstractmethod
    def load_reachability_index(self, project_id: str) -> dict[str, Any] | None:
        """Load stored Kameda reachability index."""

    @abstractmethod
    def store_partitions(
        self,
        partition_data: dict[str, Any],
        project_id: str,
    ) -> None:
        """
        Persist Fiedler partition tree.

        partition_data is the recursive tree structure with
        id, node_ids, children, fiedler_value, depth.
        """

    @abstractmethod
    def load_partitions(self, project_id: str) -> dict[str, Any] | None:
        """Load stored partition tree."""

    @abstractmethod
    def get_file_hashes(self, project_id: str) -> dict[str, str]:
        """Get stored file hashes for incremental update detection."""

    @abstractmethod
    def set_file_hashes(self, project_id: str, hashes: dict[str, str]) -> None:
        """Store file hashes for incremental update detection."""

    @abstractmethod
    def delete_nodes_by_file(self, file_path: str, project_id: str) -> int:
        """
        Delete all nodes (and their edges) belonging to a file.

        Returns the number of nodes deleted.
        """

    @abstractmethod
    def get_stats(self, project_id: str) -> dict[str, Any]:
        """Get storage statistics (node count, edge count, etc.)."""

    @abstractmethod
    def close(self) -> None:
        """Release storage resources."""


def build_graph_store(backend: str, project_path: Path) -> GraphStore:
    """
    Factory: create a GraphStore of the requested backend type.

    Args:
        backend: "sqlite" or "kuzu"
        project_path: Root path of the project being analyzed.
            Storage directory is created at project_path / .curate_ipsum /

    Returns:
        GraphStore instance

    Raises:
        ValueError: Unknown backend
        ImportError: Kuzu backend requested but kuzu not installed
    """
    storage_dir = project_path / STORAGE_DIR_NAME
    storage_dir.mkdir(parents=True, exist_ok=True)

    if backend == "sqlite":
        from curate_ipsum.storage.sqlite_graph_store import SQLiteGraphStore

        db_path = storage_dir / "graph.db"
        return SQLiteGraphStore(db_path)

    elif backend == "kuzu":
        from curate_ipsum.storage.kuzu_graph_store import KuzuGraphStore

        db_path = storage_dir / "graph.kuzu"
        return KuzuGraphStore(db_path)

    else:
        raise ValueError(f"Unknown graph store backend: {backend!r}. Supported: 'sqlite', 'kuzu'")
