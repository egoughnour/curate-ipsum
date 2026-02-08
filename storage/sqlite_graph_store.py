"""
SQLite-backed graph store.

Primary backend — zero external dependencies (stdlib sqlite3).
Uses WAL mode for concurrent read safety and batch inserts for performance.

Schema: 7 tables covering nodes, edges, Kameda labels, non-planar reachability,
partitions, partition membership, and file hashes.

Decision: D-014
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from graph.models import (
    CallGraph,
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
    FunctionSignature,
)
from storage.graph_store import GraphStore

LOG = logging.getLogger("storage.sqlite_graph_store")

_SCHEMA_SQL = """
-- Code entities
CREATE TABLE IF NOT EXISTS code_nodes (
    id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT,
    line_start INTEGER,
    line_end INTEGER,
    col_start INTEGER DEFAULT 0,
    col_end INTEGER DEFAULT 0,
    signature_json TEXT,
    docstring TEXT,
    metadata_json TEXT,
    PRIMARY KEY (id, project_id)
);

-- Relationships
CREATE TABLE IF NOT EXISTS code_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    is_conditional INTEGER DEFAULT 0,
    is_dynamic INTEGER DEFAULT 0,
    location_json TEXT,
    PRIMARY KEY (source_id, target_id, kind, project_id)
);

-- Kameda reachability labels (O(1) reachability queries)
CREATE TABLE IF NOT EXISTS kameda_labels (
    node_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    left_rank INTEGER NOT NULL,
    right_rank INTEGER NOT NULL,
    PRIMARY KEY (node_id, project_id)
);

-- Kameda index metadata (source/sink, all_node_ids)
CREATE TABLE IF NOT EXISTS kameda_meta (
    project_id TEXT PRIMARY KEY,
    source_id TEXT,
    sink_id TEXT,
    all_node_ids_json TEXT
);

-- Non-planar fallback reachability
CREATE TABLE IF NOT EXISTS nonplanar_reachability (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, project_id)
);

-- Fiedler partitions (materialized path encoding)
CREATE TABLE IF NOT EXISTS partitions (
    partition_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    depth INTEGER NOT NULL,
    fiedler_value REAL,
    node_count INTEGER,
    is_leaf INTEGER DEFAULT 0,
    PRIMARY KEY (partition_id, project_id)
);

-- Partition membership
CREATE TABLE IF NOT EXISTS partition_members (
    node_id TEXT NOT NULL,
    partition_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    PRIMARY KEY (node_id, partition_id, project_id)
);

-- File hashes for incremental update detection
CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT NOT NULL,
    project_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (file_path, project_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_nodes_project ON code_nodes(project_id);
CREATE INDEX IF NOT EXISTS idx_nodes_file ON code_nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_edges_source ON code_edges(source_id, project_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON code_edges(target_id, project_id);
CREATE INDEX IF NOT EXISTS idx_edges_project ON code_edges(project_id);
CREATE INDEX IF NOT EXISTS idx_kameda_project ON kameda_labels(project_id);
CREATE INDEX IF NOT EXISTS idx_partitions_project ON partitions(project_id);
CREATE INDEX IF NOT EXISTS idx_pmembers_partition ON partition_members(partition_id, project_id);
CREATE INDEX IF NOT EXISTS idx_file_hashes_project ON file_hashes(project_id);
"""


class SQLiteGraphStore(GraphStore):
    """SQLite-backed graph storage. Primary backend with zero external dependencies."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes (idempotent)."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ── Store / Load Graph ────────────────────────────────────────────

    def store_graph(self, graph: CallGraph, project_id: str) -> None:
        """Persist an entire call graph (bulk INSERT OR REPLACE)."""
        cur = self._conn.cursor()

        # Clear existing graph data for this project
        cur.execute("DELETE FROM code_nodes WHERE project_id = ?", (project_id,))
        cur.execute("DELETE FROM code_edges WHERE project_id = ?", (project_id,))

        # Bulk insert nodes
        node_rows = []
        for node in graph.nodes.values():
            sig_json = None
            if node.signature:
                sig_json = json.dumps({
                    "name": node.signature.name,
                    "params": list(node.signature.params),
                    "return_type": node.signature.return_type,
                    "decorators": list(node.signature.decorators),
                    "is_async": node.signature.is_async,
                    "is_generator": node.signature.is_generator,
                })
            meta_json = json.dumps(node.metadata) if node.metadata else None

            node_rows.append((
                node.id,
                project_id,
                node.kind.value,
                node.name,
                node.location.file if node.location else None,
                node.location.line_start if node.location else None,
                node.location.line_end if node.location else None,
                node.location.col_start if node.location else 0,
                node.location.col_end if node.location else 0,
                sig_json,
                node.docstring,
                meta_json,
            ))

        cur.executemany(
            "INSERT OR REPLACE INTO code_nodes "
            "(id, project_id, kind, name, file_path, line_start, line_end, "
            "col_start, col_end, signature_json, docstring, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            node_rows,
        )

        # Bulk insert edges
        edge_rows = []
        for edge in graph.edges:
            loc_json = None
            if edge.location:
                loc_json = json.dumps({
                    "file": edge.location.file,
                    "line_start": edge.location.line_start,
                    "line_end": edge.location.line_end,
                })
            edge_rows.append((
                edge.source_id,
                edge.target_id,
                project_id,
                edge.kind.value,
                edge.confidence,
                int(edge.is_conditional),
                int(edge.is_dynamic),
                loc_json,
            ))

        cur.executemany(
            "INSERT OR REPLACE INTO code_edges "
            "(source_id, target_id, project_id, kind, confidence, "
            "is_conditional, is_dynamic, location_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            edge_rows,
        )

        self._conn.commit()
        LOG.info(
            "Stored graph for project %s: %d nodes, %d edges",
            project_id, len(node_rows), len(edge_rows),
        )

    def load_graph(self, project_id: str) -> Optional[CallGraph]:
        """Load a previously stored call graph."""
        cur = self._conn.cursor()

        # Check if any data exists
        cur.execute(
            "SELECT COUNT(*) FROM code_nodes WHERE project_id = ?",
            (project_id,),
        )
        if cur.fetchone()[0] == 0:
            return None

        graph = CallGraph()

        # Load nodes
        cur.execute(
            "SELECT id, kind, name, file_path, line_start, line_end, "
            "col_start, col_end, signature_json, docstring, metadata_json "
            "FROM code_nodes WHERE project_id = ?",
            (project_id,),
        )
        for row in cur.fetchall():
            (nid, kind, name, file_path, line_start, line_end,
             col_start, col_end, sig_json, docstring, meta_json) = row

            location = None
            if file_path and line_start is not None:
                location = SourceLocation(
                    file=file_path,
                    line_start=line_start,
                    line_end=line_end or line_start,
                    col_start=col_start or 0,
                    col_end=col_end or 0,
                )

            signature = None
            if sig_json:
                sig = json.loads(sig_json)
                signature = FunctionSignature(
                    name=sig["name"],
                    params=tuple(sig.get("params", [])),
                    return_type=sig.get("return_type"),
                    decorators=tuple(sig.get("decorators", [])),
                    is_async=sig.get("is_async", False),
                    is_generator=sig.get("is_generator", False),
                )

            metadata = json.loads(meta_json) if meta_json else {}

            graph.add_node(GraphNode(
                id=nid,
                kind=NodeKind(kind),
                name=name,
                location=location,
                signature=signature,
                docstring=docstring,
                metadata=metadata,
            ))

        # Load edges
        cur.execute(
            "SELECT source_id, target_id, kind, confidence, "
            "is_conditional, is_dynamic, location_json "
            "FROM code_edges WHERE project_id = ?",
            (project_id,),
        )
        for row in cur.fetchall():
            source_id, target_id, kind, confidence, is_cond, is_dyn, loc_json = row

            location = None
            if loc_json:
                loc = json.loads(loc_json)
                location = SourceLocation(
                    file=loc["file"],
                    line_start=loc["line_start"],
                    line_end=loc["line_end"],
                )

            graph.add_edge(GraphEdge(
                source_id=source_id,
                target_id=target_id,
                kind=EdgeKind(kind),
                location=location,
                is_conditional=bool(is_cond),
                is_dynamic=bool(is_dyn),
                confidence=confidence,
            ))

        LOG.info(
            "Loaded graph for project %s: %d nodes, %d edges",
            project_id, len(graph.nodes), len(graph.edges),
        )
        return graph

    # ── Single Node / Edge ────────────────────────────────────────────

    def store_node(self, node_data: Dict[str, Any], project_id: str) -> None:
        """Store or update a single node."""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO code_nodes "
            "(id, project_id, kind, name, file_path, line_start, line_end, "
            "col_start, col_end, signature_json, docstring, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                node_data["id"],
                project_id,
                node_data.get("kind", "function"),
                node_data.get("name", ""),
                node_data.get("file_path"),
                node_data.get("line_start"),
                node_data.get("line_end"),
                node_data.get("col_start", 0),
                node_data.get("col_end", 0),
                json.dumps(node_data["signature"]) if node_data.get("signature") else None,
                node_data.get("docstring"),
                json.dumps(node_data["metadata"]) if node_data.get("metadata") else None,
            ),
        )
        self._conn.commit()

    def store_edge(self, edge_data: Dict[str, Any], project_id: str) -> None:
        """Store or update a single edge."""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO code_edges "
            "(source_id, target_id, project_id, kind, confidence, "
            "is_conditional, is_dynamic, location_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                edge_data["source_id"],
                edge_data["target_id"],
                project_id,
                edge_data.get("kind", "calls"),
                edge_data.get("confidence", 1.0),
                int(edge_data.get("is_conditional", False)),
                int(edge_data.get("is_dynamic", False)),
                json.dumps(edge_data["location"]) if edge_data.get("location") else None,
            ),
        )
        self._conn.commit()

    def get_node(self, node_id: str, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node's data by ID."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, kind, name, file_path, line_start, line_end, "
            "signature_json, docstring, metadata_json "
            "FROM code_nodes WHERE id = ? AND project_id = ?",
            (node_id, project_id),
        )
        row = cur.fetchone()
        if not row:
            return None

        nid, kind, name, file_path, line_start, line_end, sig_json, docstring, meta_json = row
        return {
            "id": nid,
            "kind": kind,
            "name": name,
            "file_path": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "signature": json.loads(sig_json) if sig_json else None,
            "docstring": docstring,
            "metadata": json.loads(meta_json) if meta_json else {},
        }

    # ── Neighbors ─────────────────────────────────────────────────────

    def get_neighbors(
        self,
        node_id: str,
        project_id: str,
        direction: str = "outgoing",
        edge_kind: Optional[str] = None,
    ) -> List[str]:
        """Get neighboring node IDs."""
        cur = self._conn.cursor()
        results: List[str] = []

        if direction in ("outgoing", "both"):
            if edge_kind:
                cur.execute(
                    "SELECT target_id FROM code_edges "
                    "WHERE source_id = ? AND project_id = ? AND kind = ?",
                    (node_id, project_id, edge_kind),
                )
            else:
                cur.execute(
                    "SELECT target_id FROM code_edges "
                    "WHERE source_id = ? AND project_id = ?",
                    (node_id, project_id),
                )
            results.extend(row[0] for row in cur.fetchall())

        if direction in ("incoming", "both"):
            if edge_kind:
                cur.execute(
                    "SELECT source_id FROM code_edges "
                    "WHERE target_id = ? AND project_id = ? AND kind = ?",
                    (node_id, project_id, edge_kind),
                )
            else:
                cur.execute(
                    "SELECT source_id FROM code_edges "
                    "WHERE target_id = ? AND project_id = ?",
                    (node_id, project_id),
                )
            results.extend(row[0] for row in cur.fetchall())

        return results

    # ── Reachability Index ────────────────────────────────────────────

    def query_reachable(
        self, source_id: str, target_id: str, project_id: str
    ) -> bool:
        """Check if target is reachable from source using Kameda labels."""
        cur = self._conn.cursor()

        # First try Kameda O(1) lookup
        cur.execute(
            "SELECT left_rank, right_rank FROM kameda_labels "
            "WHERE node_id = ? AND project_id = ?",
            (source_id, project_id),
        )
        src_row = cur.fetchone()

        cur.execute(
            "SELECT left_rank, right_rank FROM kameda_labels "
            "WHERE node_id = ? AND project_id = ?",
            (target_id, project_id),
        )
        tgt_row = cur.fetchone()

        if src_row and tgt_row:
            src_left, src_right = src_row
            tgt_left, tgt_right = tgt_row
            if src_left <= tgt_left and src_right <= tgt_right:
                return True

        # Fallback: check non-planar reachability table
        cur.execute(
            "SELECT 1 FROM nonplanar_reachability "
            "WHERE source_id = ? AND target_id = ? AND project_id = ?",
            (source_id, target_id, project_id),
        )
        return cur.fetchone() is not None

    def store_reachability_index(
        self,
        kameda_data: Dict[str, Any],
        project_id: str,
    ) -> None:
        """Persist Kameda reachability index."""
        cur = self._conn.cursor()

        # Clear existing index for this project
        cur.execute("DELETE FROM kameda_labels WHERE project_id = ?", (project_id,))
        cur.execute("DELETE FROM kameda_meta WHERE project_id = ?", (project_id,))
        cur.execute("DELETE FROM nonplanar_reachability WHERE project_id = ?", (project_id,))

        # Store labels
        left_rank = kameda_data.get("left_rank", {})
        right_rank = kameda_data.get("right_rank", {})

        label_rows = []
        for node_id in left_rank:
            if node_id in right_rank:
                label_rows.append((
                    node_id, project_id,
                    left_rank[node_id], right_rank[node_id],
                ))

        cur.executemany(
            "INSERT INTO kameda_labels (node_id, project_id, left_rank, right_rank) "
            "VALUES (?, ?, ?, ?)",
            label_rows,
        )

        # Store metadata
        all_node_ids = kameda_data.get("all_node_ids", [])
        cur.execute(
            "INSERT INTO kameda_meta (project_id, source_id, sink_id, all_node_ids_json) "
            "VALUES (?, ?, ?, ?)",
            (
                project_id,
                kameda_data.get("source_id", ""),
                kameda_data.get("sink_id", ""),
                json.dumps(list(all_node_ids)),
            ),
        )

        # Store non-planar reachability
        np_reach = kameda_data.get("non_planar_reachability", {})
        np_rows = []
        for src, targets in np_reach.items():
            for tgt in targets:
                np_rows.append((src, tgt, project_id))

        if np_rows:
            cur.executemany(
                "INSERT OR IGNORE INTO nonplanar_reachability "
                "(source_id, target_id, project_id) VALUES (?, ?, ?)",
                np_rows,
            )

        self._conn.commit()
        LOG.info(
            "Stored Kameda index for project %s: %d labels, %d non-planar pairs",
            project_id, len(label_rows), len(np_rows),
        )

    def load_reachability_index(
        self, project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load stored Kameda reachability index."""
        cur = self._conn.cursor()

        # Check metadata exists
        cur.execute(
            "SELECT source_id, sink_id, all_node_ids_json "
            "FROM kameda_meta WHERE project_id = ?",
            (project_id,),
        )
        meta_row = cur.fetchone()
        if not meta_row:
            return None

        source_id, sink_id, all_ids_json = meta_row

        # Load labels
        cur.execute(
            "SELECT node_id, left_rank, right_rank "
            "FROM kameda_labels WHERE project_id = ?",
            (project_id,),
        )
        left_rank = {}
        right_rank = {}
        for node_id, lr, rr in cur.fetchall():
            left_rank[node_id] = lr
            right_rank[node_id] = rr

        # Load non-planar reachability
        cur.execute(
            "SELECT source_id, target_id "
            "FROM nonplanar_reachability WHERE project_id = ?",
            (project_id,),
        )
        np_reach: Dict[str, Set[str]] = {}
        for src, tgt in cur.fetchall():
            np_reach.setdefault(src, set()).add(tgt)

        return {
            "left_rank": left_rank,
            "right_rank": right_rank,
            "source_id": source_id,
            "sink_id": sink_id,
            "non_planar_reachability": np_reach,
            "all_node_ids": frozenset(json.loads(all_ids_json)) if all_ids_json else frozenset(),
        }

    # ── Partitions ────────────────────────────────────────────────────

    def store_partitions(
        self,
        partition_data: Dict[str, Any],
        project_id: str,
    ) -> None:
        """Persist Fiedler partition tree."""
        cur = self._conn.cursor()

        # Clear existing partitions for this project
        cur.execute("DELETE FROM partitions WHERE project_id = ?", (project_id,))
        cur.execute("DELETE FROM partition_members WHERE project_id = ?", (project_id,))

        # Recursively store partition tree
        part_rows: List[Tuple] = []
        member_rows: List[Tuple] = []
        self._flatten_partition(partition_data, project_id, part_rows, member_rows)

        cur.executemany(
            "INSERT OR REPLACE INTO partitions "
            "(partition_id, project_id, depth, fiedler_value, node_count, is_leaf) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            part_rows,
        )
        cur.executemany(
            "INSERT OR REPLACE INTO partition_members "
            "(node_id, partition_id, project_id) VALUES (?, ?, ?)",
            member_rows,
        )

        self._conn.commit()
        LOG.info(
            "Stored partitions for project %s: %d partitions, %d memberships",
            project_id, len(part_rows), len(member_rows),
        )

    def _flatten_partition(
        self,
        pdata: Dict[str, Any],
        project_id: str,
        part_rows: List[Tuple],
        member_rows: List[Tuple],
    ) -> None:
        """Recursively flatten partition tree into rows."""
        node_ids = pdata.get("node_ids", [])
        children = pdata.get("children")
        is_leaf = 1 if children is None else 0

        part_rows.append((
            pdata["id"],
            project_id,
            pdata.get("depth", 0),
            pdata.get("fiedler_value"),
            len(node_ids),
            is_leaf,
        ))

        # Store direct member nodes (only for leaf partitions to avoid duplication)
        if is_leaf:
            for nid in node_ids:
                member_rows.append((nid, pdata["id"], project_id))

        # Recurse into children
        if children:
            for child in children:
                self._flatten_partition(child, project_id, part_rows, member_rows)

    def load_partitions(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Load stored partition tree."""
        cur = self._conn.cursor()

        cur.execute(
            "SELECT partition_id, depth, fiedler_value, node_count, is_leaf "
            "FROM partitions WHERE project_id = ? ORDER BY partition_id",
            (project_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return None

        # Load all memberships
        cur.execute(
            "SELECT node_id, partition_id "
            "FROM partition_members WHERE project_id = ?",
            (project_id,),
        )
        memberships: Dict[str, List[str]] = {}
        for nid, pid in cur.fetchall():
            memberships.setdefault(pid, []).append(nid)

        # Reconstruct tree from flat rows
        partitions: Dict[str, Dict[str, Any]] = {}
        for pid, depth, fiedler, node_count, is_leaf in rows:
            partitions[pid] = {
                "id": pid,
                "depth": depth,
                "fiedler_value": fiedler,
                "node_ids": memberships.get(pid, []),
                "children": None if is_leaf else [],
            }

        # Link children by partition ID convention (parent "0", children "0.0", "0.1")
        for pid, pdata in partitions.items():
            if pdata["children"] is not None:
                # Find direct children: IDs that are pid + ".0" and pid + ".1"
                left_id = f"{pid}.0"
                right_id = f"{pid}.1"
                children = []
                if left_id in partitions:
                    children.append(partitions[left_id])
                if right_id in partitions:
                    children.append(partitions[right_id])
                if children:
                    pdata["children"] = children
                else:
                    pdata["children"] = None

        # Propagate node_ids from leaves to parents
        self._propagate_node_ids(partitions)

        # Return root partition
        root = partitions.get("0")
        return root

    @staticmethod
    def _propagate_node_ids(partitions: Dict[str, Dict[str, Any]]) -> None:
        """Propagate node_ids from leaf partitions up to parents."""
        # Process deepest first
        by_depth = sorted(partitions.values(), key=lambda p: p["depth"], reverse=True)
        for pdata in by_depth:
            if pdata["children"]:
                all_ids = set()
                for child in pdata["children"]:
                    all_ids.update(child.get("node_ids", []))
                pdata["node_ids"] = list(all_ids)

    # ── File Hashes ───────────────────────────────────────────────────

    def get_file_hashes(self, project_id: str) -> Dict[str, str]:
        """Get stored file hashes for incremental update detection."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT file_path, content_hash FROM file_hashes WHERE project_id = ?",
            (project_id,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def set_file_hashes(
        self, project_id: str, hashes: Dict[str, str]
    ) -> None:
        """Store file hashes for incremental update detection."""
        cur = self._conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        # Upsert each hash
        rows = [
            (fp, project_id, h, now)
            for fp, h in hashes.items()
        ]
        cur.executemany(
            "INSERT OR REPLACE INTO file_hashes "
            "(file_path, project_id, content_hash, last_updated) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    # ── Delete by File ────────────────────────────────────────────────

    def delete_nodes_by_file(self, file_path: str, project_id: str) -> int:
        """Delete all nodes (and their edges) belonging to a file."""
        cur = self._conn.cursor()

        # Get node IDs in this file
        cur.execute(
            "SELECT id FROM code_nodes WHERE file_path = ? AND project_id = ?",
            (file_path, project_id),
        )
        node_ids = [row[0] for row in cur.fetchall()]
        if not node_ids:
            return 0

        # Delete edges involving these nodes
        placeholders = ",".join("?" * len(node_ids))
        params = node_ids + [project_id]
        cur.execute(
            f"DELETE FROM code_edges WHERE project_id = ? AND "
            f"(source_id IN ({placeholders}) OR target_id IN ({placeholders}))",
            [project_id] + node_ids + node_ids,
        )

        # Delete the nodes
        cur.execute(
            f"DELETE FROM code_nodes WHERE project_id = ? AND id IN ({placeholders})",
            [project_id] + node_ids,
        )

        # Delete from Kameda labels
        cur.execute(
            f"DELETE FROM kameda_labels WHERE project_id = ? AND node_id IN ({placeholders})",
            [project_id] + node_ids,
        )

        # Delete file hash
        cur.execute(
            "DELETE FROM file_hashes WHERE file_path = ? AND project_id = ?",
            (file_path, project_id),
        )

        self._conn.commit()
        LOG.info("Deleted %d nodes for file %s (project %s)", len(node_ids), file_path, project_id)
        return len(node_ids)

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self, project_id: str) -> Dict[str, Any]:
        """Get storage statistics."""
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) FROM code_nodes WHERE project_id = ?", (project_id,))
        node_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM code_edges WHERE project_id = ?", (project_id,))
        edge_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM kameda_labels WHERE project_id = ?", (project_id,))
        kameda_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM partitions WHERE project_id = ?", (project_id,))
        partition_count = cur.fetchone()[0]

        cur.execute(
            "SELECT MAX(last_updated) FROM file_hashes WHERE project_id = ?",
            (project_id,),
        )
        last_updated = cur.fetchone()[0]

        return {
            "backend": "sqlite",
            "node_count": node_count,
            "edge_count": edge_count,
            "has_kameda_index": kameda_count > 0,
            "kameda_label_count": kameda_count,
            "has_partitions": partition_count > 0,
            "partition_count": partition_count,
            "last_updated": last_updated,
            "db_path": str(self._db_path),
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Release database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
