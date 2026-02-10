"""
Kuzu-backed graph store.

Optional backend — requires ``pip install kuzu``.
Provides native Cypher query support and efficient multi-hop traversals.

Uses the same GraphStore ABC as the SQLite backend (D-014 pattern).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC
from pathlib import Path
from typing import Any

from curate_ipsum.graph.models import (
    CallGraph,
    EdgeKind,
    FunctionSignature,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
)
from curate_ipsum.storage.graph_store import GraphStore

LOG = logging.getLogger("storage.kuzu_graph_store")

try:
    import kuzu
except ImportError:
    kuzu = None  # type: ignore


# Schema DDL (Cypher CREATE TABLE statements)
_NODE_TABLES = [
    """
    CREATE NODE TABLE IF NOT EXISTS CodeNode(
        id STRING,
        project_id STRING,
        kind STRING,
        name STRING,
        file_path STRING,
        line_start INT64,
        line_end INT64,
        signature_json STRING,
        docstring STRING,
        metadata_json STRING,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS KamedaLabel(
        id STRING,
        project_id STRING,
        left_rank INT64,
        right_rank INT64,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS PartitionNode(
        id STRING,
        project_id STRING,
        depth INT64,
        fiedler_value DOUBLE,
        node_count INT64,
        is_leaf BOOLEAN,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS FileHash(
        id STRING,
        project_id STRING,
        content_hash STRING,
        last_updated STRING,
        PRIMARY KEY (id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS KamedaMeta(
        id STRING,
        source_id STRING,
        sink_id STRING,
        all_node_ids_json STRING,
        PRIMARY KEY (id)
    )
    """,
]

_REL_TABLES = [
    """
    CREATE REL TABLE IF NOT EXISTS CALLS(
        FROM CodeNode TO CodeNode,
        project_id STRING,
        confidence DOUBLE DEFAULT 1.0,
        is_conditional BOOLEAN DEFAULT FALSE,
        is_dynamic BOOLEAN DEFAULT FALSE,
        location_json STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS DEFINES(
        FROM CodeNode TO CodeNode,
        project_id STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS INHERITS(
        FROM CodeNode TO CodeNode,
        project_id STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS IMPORTS(
        FROM CodeNode TO CodeNode,
        project_id STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS REFERENCES_EDGE(
        FROM CodeNode TO CodeNode,
        project_id STRING,
        confidence DOUBLE DEFAULT 1.0
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS NONPLANAR_REACH(
        FROM CodeNode TO CodeNode,
        project_id STRING
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS PARTITION_MEMBER(
        FROM CodeNode TO PartitionNode,
        project_id STRING
    )
    """,
]

# Map EdgeKind to Kuzu relationship table names
_EDGE_KIND_TO_REL = {
    EdgeKind.CALLS: "CALLS",
    EdgeKind.DEFINES: "DEFINES",
    EdgeKind.INHERITS: "INHERITS",
    EdgeKind.IMPORTS: "IMPORTS",
    EdgeKind.REFERENCES: "REFERENCES_EDGE",
}


class KuzuGraphStore(GraphStore):
    """Kuzu-backed graph storage with native Cypher support."""

    def __init__(self, db_path: Path) -> None:
        if kuzu is None:
            raise ImportError("kuzu required for KuzuGraphStore. Install with: pip install 'curate-ipsum[graphdb]'")
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create node and relationship tables (idempotent)."""
        for ddl in _NODE_TABLES:
            try:
                self._conn.execute(ddl.strip())
            except Exception as exc:
                # Table already exists — safe to ignore
                if "already exists" not in str(exc).lower():
                    LOG.warning("Kuzu schema warning: %s", exc)

        for ddl in _REL_TABLES:
            try:
                self._conn.execute(ddl.strip())
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    LOG.warning("Kuzu schema warning: %s", exc)

    # ── Store / Load Graph ────────────────────────────────────────────

    def store_graph(self, graph: CallGraph, project_id: str) -> None:
        """Persist an entire call graph."""
        # Clear existing data for this project
        self._clear_project_nodes(project_id)

        # Insert nodes
        for node in graph.nodes.values():
            sig_json = None
            if node.signature:
                sig_json = json.dumps(
                    {
                        "name": node.signature.name,
                        "params": list(node.signature.params),
                        "return_type": node.signature.return_type,
                        "decorators": list(node.signature.decorators),
                        "is_async": node.signature.is_async,
                        "is_generator": node.signature.is_generator,
                    }
                )
            meta_json = json.dumps(node.metadata) if node.metadata else None

            self._conn.execute(
                "CREATE (n:CodeNode {"
                "id: $id, project_id: $pid, kind: $kind, name: $name, "
                "file_path: $fp, line_start: $ls, line_end: $le, "
                "signature_json: $sig, docstring: $doc, metadata_json: $meta"
                "})",
                {
                    "id": node.id,
                    "pid": project_id,
                    "kind": node.kind.value,
                    "name": node.name,
                    "fp": node.location.file if node.location else "",
                    "ls": node.location.line_start if node.location else 0,
                    "le": node.location.line_end if node.location else 0,
                    "sig": sig_json or "",
                    "doc": node.docstring or "",
                    "meta": meta_json or "",
                },
            )

        # Insert edges
        for edge in graph.edges:
            rel_name = _EDGE_KIND_TO_REL.get(edge.kind, "CALLS")
            loc_json = ""
            if edge.location:
                loc_json = json.dumps(
                    {
                        "file": edge.location.file,
                        "line_start": edge.location.line_start,
                        "line_end": edge.location.line_end,
                    }
                )

            try:
                if rel_name == "CALLS":
                    self._conn.execute(
                        "MATCH (a:CodeNode {id: $src}), (b:CodeNode {id: $tgt}) "
                        "CREATE (a)-[:CALLS {"
                        "project_id: $pid, confidence: $conf, "
                        "is_conditional: $cond, is_dynamic: $dyn, "
                        "location_json: $loc"
                        "}]->(b)",
                        {
                            "src": edge.source_id,
                            "tgt": edge.target_id,
                            "pid": project_id,
                            "conf": edge.confidence,
                            "cond": edge.is_conditional,
                            "dyn": edge.is_dynamic,
                            "loc": loc_json,
                        },
                    )
                elif rel_name == "REFERENCES_EDGE":
                    self._conn.execute(
                        "MATCH (a:CodeNode {id: $src}), (b:CodeNode {id: $tgt}) "
                        "CREATE (a)-[:REFERENCES_EDGE {"
                        "project_id: $pid, confidence: $conf"
                        "}]->(b)",
                        {
                            "src": edge.source_id,
                            "tgt": edge.target_id,
                            "pid": project_id,
                            "conf": edge.confidence,
                        },
                    )
                else:
                    self._conn.execute(
                        f"MATCH (a:CodeNode {{id: $src}}), (b:CodeNode {{id: $tgt}}) "
                        f"CREATE (a)-[:{rel_name} {{project_id: $pid}}]->(b)",
                        {
                            "src": edge.source_id,
                            "tgt": edge.target_id,
                            "pid": project_id,
                        },
                    )
            except Exception as exc:
                LOG.debug("Failed to create edge %s -> %s: %s", edge.source_id, edge.target_id, exc)

        LOG.info(
            "Stored graph for project %s: %d nodes, %d edges",
            project_id,
            len(graph.nodes),
            len(graph.edges),
        )

    def load_graph(self, project_id: str) -> CallGraph | None:
        """Load a previously stored call graph."""
        # Check if data exists
        result = self._conn.execute(
            "MATCH (n:CodeNode {project_id: $pid}) RETURN count(n)",
            {"pid": project_id},
        )
        count = 0
        while result.has_next():
            count = result.get_next()[0]
        if count == 0:
            return None

        graph = CallGraph()

        # Load nodes
        result = self._conn.execute(
            "MATCH (n:CodeNode {project_id: $pid}) "
            "RETURN n.id, n.kind, n.name, n.file_path, n.line_start, n.line_end, "
            "n.signature_json, n.docstring, n.metadata_json",
            {"pid": project_id},
        )
        while result.has_next():
            row = result.get_next()
            nid, kind, name, file_path, line_start, line_end, sig_json, docstring, meta_json = row

            location = None
            if file_path and line_start:
                location = SourceLocation(
                    file=file_path,
                    line_start=line_start,
                    line_end=line_end or line_start,
                )

            signature = None
            if sig_json:
                try:
                    sig = json.loads(sig_json)
                    signature = FunctionSignature(
                        name=sig["name"],
                        params=tuple(sig.get("params", [])),
                        return_type=sig.get("return_type"),
                        decorators=tuple(sig.get("decorators", [])),
                        is_async=sig.get("is_async", False),
                        is_generator=sig.get("is_generator", False),
                    )
                except (json.JSONDecodeError, KeyError):
                    pass

            metadata = {}
            if meta_json:
                try:
                    metadata = json.loads(meta_json)
                except json.JSONDecodeError:
                    pass

            graph.add_node(
                GraphNode(
                    id=nid,
                    kind=NodeKind(kind),
                    name=name,
                    location=location,
                    signature=signature,
                    docstring=docstring,
                    metadata=metadata,
                )
            )

        # Load edges for each relationship type
        for edge_kind, rel_name in _EDGE_KIND_TO_REL.items():
            try:
                if rel_name in ("CALLS",):
                    result = self._conn.execute(
                        f"MATCH (a:CodeNode)-[r:{rel_name} {{project_id: $pid}}]->(b:CodeNode) "
                        f"RETURN a.id, b.id, r.confidence, r.is_conditional, r.is_dynamic, r.location_json",
                        {"pid": project_id},
                    )
                    while result.has_next():
                        row = result.get_next()
                        src, tgt, conf, cond, dyn, loc_json = row
                        location = None
                        if loc_json:
                            try:
                                loc = json.loads(loc_json)
                                location = SourceLocation(
                                    file=loc["file"],
                                    line_start=loc["line_start"],
                                    line_end=loc["line_end"],
                                )
                            except (json.JSONDecodeError, KeyError):
                                pass
                        graph.add_edge(
                            GraphEdge(
                                source_id=src,
                                target_id=tgt,
                                kind=edge_kind,
                                confidence=conf or 1.0,
                                is_conditional=bool(cond),
                                is_dynamic=bool(dyn),
                                location=location,
                            )
                        )
                elif rel_name == "REFERENCES_EDGE":
                    result = self._conn.execute(
                        f"MATCH (a:CodeNode)-[r:{rel_name} {{project_id: $pid}}]->(b:CodeNode) "
                        f"RETURN a.id, b.id, r.confidence",
                        {"pid": project_id},
                    )
                    while result.has_next():
                        row = result.get_next()
                        src, tgt, conf = row
                        graph.add_edge(
                            GraphEdge(
                                source_id=src,
                                target_id=tgt,
                                kind=edge_kind,
                                confidence=conf or 1.0,
                            )
                        )
                else:
                    result = self._conn.execute(
                        f"MATCH (a:CodeNode)-[r:{rel_name} {{project_id: $pid}}]->(b:CodeNode) RETURN a.id, b.id",
                        {"pid": project_id},
                    )
                    while result.has_next():
                        row = result.get_next()
                        src, tgt = row
                        graph.add_edge(
                            GraphEdge(
                                source_id=src,
                                target_id=tgt,
                                kind=edge_kind,
                            )
                        )
            except Exception as exc:
                LOG.debug("Failed to load %s edges: %s", rel_name, exc)

        LOG.info(
            "Loaded graph for project %s: %d nodes, %d edges",
            project_id,
            len(graph.nodes),
            len(graph.edges),
        )
        return graph

    # ── Single Node / Edge ────────────────────────────────────────────

    def store_node(self, node_data: dict[str, Any], project_id: str) -> None:
        """Store or update a single node."""
        # Delete if exists, then create
        try:
            self._conn.execute(
                "MATCH (n:CodeNode {id: $id}) DELETE n",
                {"id": node_data["id"]},
            )
        except Exception:
            pass

        self._conn.execute(
            "CREATE (n:CodeNode {"
            "id: $id, project_id: $pid, kind: $kind, name: $name, "
            "file_path: $fp, line_start: $ls, line_end: $le, "
            "signature_json: $sig, docstring: $doc, metadata_json: $meta"
            "})",
            {
                "id": node_data["id"],
                "pid": project_id,
                "kind": node_data.get("kind", "function"),
                "name": node_data.get("name", ""),
                "fp": node_data.get("file_path", ""),
                "ls": node_data.get("line_start", 0),
                "le": node_data.get("line_end", 0),
                "sig": json.dumps(node_data["signature"]) if node_data.get("signature") else "",
                "doc": node_data.get("docstring", ""),
                "meta": json.dumps(node_data["metadata"]) if node_data.get("metadata") else "",
            },
        )

    def store_edge(self, edge_data: dict[str, Any], project_id: str) -> None:
        """Store or update a single edge."""
        kind_str = edge_data.get("kind", "calls")
        try:
            edge_kind = EdgeKind(kind_str)
        except ValueError:
            edge_kind = EdgeKind.CALLS
        rel_name = _EDGE_KIND_TO_REL.get(edge_kind, "CALLS")

        try:
            self._conn.execute(
                f"MATCH (a:CodeNode {{id: $src}}), (b:CodeNode {{id: $tgt}}) "
                f"CREATE (a)-[:{rel_name} {{project_id: $pid}}]->(b)",
                {
                    "src": edge_data["source_id"],
                    "tgt": edge_data["target_id"],
                    "pid": project_id,
                },
            )
        except Exception as exc:
            LOG.debug("Failed to store edge: %s", exc)

    def get_node(self, node_id: str, project_id: str) -> dict[str, Any] | None:
        """Get a single node's data by ID."""
        result = self._conn.execute(
            "MATCH (n:CodeNode {id: $id, project_id: $pid}) "
            "RETURN n.id, n.kind, n.name, n.file_path, n.line_start, n.line_end, "
            "n.signature_json, n.docstring, n.metadata_json",
            {"id": node_id, "pid": project_id},
        )
        if result.has_next():
            row = result.get_next()
            nid, kind, name, fp, ls, le, sig_json, doc, meta_json = row
            return {
                "id": nid,
                "kind": kind,
                "name": name,
                "file_path": fp,
                "line_start": ls,
                "line_end": le,
                "signature": json.loads(sig_json) if sig_json else None,
                "docstring": doc,
                "metadata": json.loads(meta_json) if meta_json else {},
            }
        return None

    # ── Neighbors ─────────────────────────────────────────────────────

    def get_neighbors(
        self,
        node_id: str,
        project_id: str,
        direction: str = "outgoing",
        edge_kind: str | None = None,
    ) -> list[str]:
        """Get neighboring node IDs."""
        results: list[str] = []

        if edge_kind:
            try:
                ek = EdgeKind(edge_kind)
            except ValueError:
                return results
            rel_name = _EDGE_KIND_TO_REL.get(ek, "CALLS")
            rel_names = [rel_name]
        else:
            rel_names = list(_EDGE_KIND_TO_REL.values())

        for rel in rel_names:
            try:
                if direction in ("outgoing", "both"):
                    r = self._conn.execute(
                        f"MATCH (a:CodeNode {{id: $id}})-[:{rel} {{project_id: $pid}}]->(b:CodeNode) RETURN b.id",
                        {"id": node_id, "pid": project_id},
                    )
                    while r.has_next():
                        results.append(r.get_next()[0])

                if direction in ("incoming", "both"):
                    r = self._conn.execute(
                        f"MATCH (a:CodeNode)-[:{rel} {{project_id: $pid}}]->(b:CodeNode {{id: $id}}) RETURN a.id",
                        {"id": node_id, "pid": project_id},
                    )
                    while r.has_next():
                        results.append(r.get_next()[0])
            except Exception as exc:
                LOG.debug("Neighbor query failed for %s: %s", rel, exc)

        return results

    # ── Reachability Index ────────────────────────────────────────────

    def query_reachable(self, source_id: str, target_id: str, project_id: str) -> bool:
        """Check if target is reachable from source using Kameda labels."""
        # First try Kameda labels
        src_label = self._get_kameda_label(source_id, project_id)
        tgt_label = self._get_kameda_label(target_id, project_id)

        if src_label and tgt_label:
            if src_label[0] <= tgt_label[0] and src_label[1] <= tgt_label[1]:
                return True

        # Fallback: check non-planar reachability
        try:
            result = self._conn.execute(
                "MATCH (a:CodeNode {id: $src})-[:NONPLANAR_REACH {project_id: $pid}]->(b:CodeNode {id: $tgt}) "
                "RETURN count(*)",
                {"src": source_id, "tgt": target_id, "pid": project_id},
            )
            if result.has_next():
                return result.get_next()[0] > 0
        except Exception:
            pass

        return False

    def _get_kameda_label(self, node_id: str, project_id: str) -> tuple | None:
        """Get (left_rank, right_rank) for a node."""
        # Kameda labels stored with composite key: project_id::node_id
        label_id = f"{project_id}::{node_id}"
        try:
            result = self._conn.execute(
                "MATCH (k:KamedaLabel {id: $id}) RETURN k.left_rank, k.right_rank",
                {"id": label_id},
            )
            if result.has_next():
                row = result.get_next()
                return (row[0], row[1])
        except Exception:
            pass
        return None

    def store_reachability_index(
        self,
        kameda_data: dict[str, Any],
        project_id: str,
    ) -> None:
        """Persist Kameda reachability index."""
        # Clear existing labels
        try:
            self._conn.execute(
                "MATCH (k:KamedaLabel {project_id: $pid}) DELETE k",
                {"pid": project_id},
            )
            self._conn.execute(
                "MATCH (m:KamedaMeta {id: $pid}) DELETE m",
                {"pid": project_id},
            )
        except Exception:
            pass

        # Store labels
        left_rank = kameda_data.get("left_rank", {})
        right_rank = kameda_data.get("right_rank", {})

        for node_id in left_rank:
            if node_id in right_rank:
                label_id = f"{project_id}::{node_id}"
                self._conn.execute(
                    "CREATE (k:KamedaLabel {id: $id, project_id: $pid, left_rank: $lr, right_rank: $rr})",
                    {
                        "id": label_id,
                        "pid": project_id,
                        "lr": left_rank[node_id],
                        "rr": right_rank[node_id],
                    },
                )

        # Store metadata
        all_ids = kameda_data.get("all_node_ids", [])
        self._conn.execute(
            "CREATE (m:KamedaMeta {id: $pid, source_id: $src, sink_id: $sink, all_node_ids_json: $ids})",
            {
                "pid": project_id,
                "src": kameda_data.get("source_id", ""),
                "sink": kameda_data.get("sink_id", ""),
                "ids": json.dumps(list(all_ids)),
            },
        )

        # Store non-planar reachability as edges
        np_reach = kameda_data.get("non_planar_reachability", {})
        for src, targets in np_reach.items():
            for tgt in targets:
                try:
                    self._conn.execute(
                        "MATCH (a:CodeNode {id: $src}), (b:CodeNode {id: $tgt}) "
                        "CREATE (a)-[:NONPLANAR_REACH {project_id: $pid}]->(b)",
                        {"src": src, "tgt": tgt, "pid": project_id},
                    )
                except Exception:
                    pass

        LOG.info(
            "Stored Kameda index for project %s: %d labels",
            project_id,
            len(left_rank),
        )

    def load_reachability_index(self, project_id: str) -> dict[str, Any] | None:
        """Load stored Kameda reachability index."""
        # Check metadata exists
        try:
            result = self._conn.execute(
                "MATCH (m:KamedaMeta {id: $pid}) RETURN m.source_id, m.sink_id, m.all_node_ids_json",
                {"pid": project_id},
            )
            if not result.has_next():
                return None
            row = result.get_next()
            source_id, sink_id, all_ids_json = row
        except Exception:
            return None

        # Load labels
        left_rank = {}
        right_rank = {}
        try:
            result = self._conn.execute(
                "MATCH (k:KamedaLabel {project_id: $pid}) RETURN k.id, k.left_rank, k.right_rank",
                {"pid": project_id},
            )
            while result.has_next():
                row = result.get_next()
                label_id, lr, rr = row
                # Extract node_id from composite key
                node_id = label_id.split("::", 1)[1] if "::" in label_id else label_id
                left_rank[node_id] = lr
                right_rank[node_id] = rr
        except Exception:
            pass

        # Load non-planar reachability
        np_reach: dict[str, set[str]] = {}
        try:
            result = self._conn.execute(
                "MATCH (a:CodeNode)-[:NONPLANAR_REACH {project_id: $pid}]->(b:CodeNode) RETURN a.id, b.id",
                {"pid": project_id},
            )
            while result.has_next():
                row = result.get_next()
                np_reach.setdefault(row[0], set()).add(row[1])
        except Exception:
            pass

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
        partition_data: dict[str, Any],
        project_id: str,
    ) -> None:
        """Persist Fiedler partition tree."""
        # Clear existing
        try:
            self._conn.execute(
                "MATCH (p:PartitionNode {project_id: $pid}) DELETE p",
                {"pid": project_id},
            )
        except Exception:
            pass

        # Flatten and store
        self._store_partition_recursive(partition_data, project_id)
        LOG.info("Stored partitions for project %s", project_id)

    def _store_partition_recursive(self, pdata: dict[str, Any], project_id: str) -> None:
        """Recursively store partition nodes."""
        children = pdata.get("children")
        is_leaf = children is None
        node_ids = pdata.get("node_ids", [])

        part_id = f"{project_id}::{pdata['id']}"
        self._conn.execute(
            "CREATE (p:PartitionNode {"
            "id: $id, project_id: $pid, depth: $d, "
            "fiedler_value: $fv, node_count: $nc, is_leaf: $leaf"
            "})",
            {
                "id": part_id,
                "pid": project_id,
                "d": pdata.get("depth", 0),
                "fv": pdata.get("fiedler_value", 0.0) or 0.0,
                "nc": len(node_ids),
                "leaf": is_leaf,
            },
        )

        # Store membership for leaf partitions
        if is_leaf:
            for nid in node_ids:
                try:
                    self._conn.execute(
                        "MATCH (c:CodeNode {id: $nid}), (p:PartitionNode {id: $pid}) "
                        "CREATE (c)-[:PARTITION_MEMBER {project_id: $proj}]->(p)",
                        {"nid": nid, "pid": part_id, "proj": project_id},
                    )
                except Exception:
                    pass

        if children:
            for child in children:
                self._store_partition_recursive(child, project_id)

    def load_partitions(self, project_id: str) -> dict[str, Any] | None:
        """Load stored partition tree."""
        try:
            result = self._conn.execute(
                "MATCH (p:PartitionNode {project_id: $pid}) "
                "RETURN p.id, p.depth, p.fiedler_value, p.node_count, p.is_leaf "
                "ORDER BY p.id",
                {"pid": project_id},
            )
        except Exception:
            return None

        partitions: dict[str, dict[str, Any]] = {}
        while result.has_next():
            row = result.get_next()
            full_id, depth, fiedler, node_count, is_leaf = row
            # Extract partition ID from composite key
            pid = full_id.split("::", 1)[1] if "::" in full_id else full_id

            # Load members for leaf partitions
            members = []
            if is_leaf:
                try:
                    mr = self._conn.execute(
                        "MATCH (c:CodeNode)-[:PARTITION_MEMBER {project_id: $proj}]->(p:PartitionNode {id: $pid}) "
                        "RETURN c.id",
                        {"proj": project_id, "pid": full_id},
                    )
                    while mr.has_next():
                        members.append(mr.get_next()[0])
                except Exception:
                    pass

            partitions[pid] = {
                "id": pid,
                "depth": depth,
                "fiedler_value": fiedler,
                "node_ids": members,
                "children": None if is_leaf else [],
            }

        if not partitions:
            return None

        # Link children
        for pid, pdata in partitions.items():
            if pdata["children"] is not None:
                left_id = f"{pid}.0"
                right_id = f"{pid}.1"
                children = []
                if left_id in partitions:
                    children.append(partitions[left_id])
                if right_id in partitions:
                    children.append(partitions[right_id])
                pdata["children"] = children or None

        # Propagate node_ids
        by_depth = sorted(partitions.values(), key=lambda p: p["depth"], reverse=True)
        for pdata in by_depth:
            if pdata["children"]:
                all_ids = set()
                for child in pdata["children"]:
                    all_ids.update(child.get("node_ids", []))
                pdata["node_ids"] = list(all_ids)

        return partitions.get("0")

    # ── File Hashes ───────────────────────────────────────────────────

    def get_file_hashes(self, project_id: str) -> dict[str, str]:
        """Get stored file hashes for incremental update detection."""
        result_dict: dict[str, str] = {}
        try:
            result = self._conn.execute(
                "MATCH (f:FileHash {project_id: $pid}) RETURN f.id, f.content_hash",
                {"pid": project_id},
            )
            while result.has_next():
                row = result.get_next()
                # Extract file_path from composite key
                file_path = row[0].split("::", 1)[1] if "::" in row[0] else row[0]
                result_dict[file_path] = row[1]
        except Exception:
            pass
        return result_dict

    def set_file_hashes(self, project_id: str, hashes: dict[str, str]) -> None:
        """Store file hashes for incremental update detection."""
        from datetime import datetime

        now = datetime.now(UTC).isoformat()

        for fp, h in hashes.items():
            fh_id = f"{project_id}::{fp}"
            # Delete existing
            try:
                self._conn.execute(
                    "MATCH (f:FileHash {id: $id}) DELETE f",
                    {"id": fh_id},
                )
            except Exception:
                pass
            # Create new
            self._conn.execute(
                "CREATE (f:FileHash {id: $id, project_id: $pid, content_hash: $h, last_updated: $ts})",
                {"id": fh_id, "pid": project_id, "h": h, "ts": now},
            )

    # ── Delete by File ────────────────────────────────────────────────

    def delete_nodes_by_file(self, file_path: str, project_id: str) -> int:
        """Delete all nodes (and their edges) belonging to a file."""
        count = 0
        try:
            # Count first
            result = self._conn.execute(
                "MATCH (n:CodeNode {file_path: $fp, project_id: $pid}) RETURN count(n)",
                {"fp": file_path, "pid": project_id},
            )
            if result.has_next():
                count = result.get_next()[0]

            if count > 0:
                # Delete nodes (Kuzu cascades relationship deletions)
                self._conn.execute(
                    "MATCH (n:CodeNode {file_path: $fp, project_id: $pid}) DETACH DELETE n",
                    {"fp": file_path, "pid": project_id},
                )

            # Delete file hash
            fh_id = f"{project_id}::{file_path}"
            try:
                self._conn.execute(
                    "MATCH (f:FileHash {id: $id}) DELETE f",
                    {"id": fh_id},
                )
            except Exception:
                pass

        except Exception as exc:
            LOG.debug("Failed to delete nodes for file %s: %s", file_path, exc)

        return count

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self, project_id: str) -> dict[str, Any]:
        """Get storage statistics."""
        stats: dict[str, Any] = {"backend": "kuzu"}

        try:
            r = self._conn.execute(
                "MATCH (n:CodeNode {project_id: $pid}) RETURN count(n)",
                {"pid": project_id},
            )
            stats["node_count"] = r.get_next()[0] if r.has_next() else 0
        except Exception:
            stats["node_count"] = 0

        # Count all edge types
        edge_count = 0
        for rel in _EDGE_KIND_TO_REL.values():
            try:
                r = self._conn.execute(
                    f"MATCH ()-[r:{rel} {{project_id: $pid}}]->() RETURN count(r)",
                    {"pid": project_id},
                )
                if r.has_next():
                    edge_count += r.get_next()[0]
            except Exception:
                pass
        stats["edge_count"] = edge_count

        try:
            r = self._conn.execute(
                "MATCH (k:KamedaLabel {project_id: $pid}) RETURN count(k)",
                {"pid": project_id},
            )
            kameda_count = r.get_next()[0] if r.has_next() else 0
            stats["has_kameda_index"] = kameda_count > 0
            stats["kameda_label_count"] = kameda_count
        except Exception:
            stats["has_kameda_index"] = False
            stats["kameda_label_count"] = 0

        try:
            r = self._conn.execute(
                "MATCH (p:PartitionNode {project_id: $pid}) RETURN count(p)",
                {"pid": project_id},
            )
            part_count = r.get_next()[0] if r.has_next() else 0
            stats["has_partitions"] = part_count > 0
            stats["partition_count"] = part_count
        except Exception:
            stats["has_partitions"] = False
            stats["partition_count"] = 0

        stats["db_path"] = str(self._db_path)
        return stats

    # ── Helpers ───────────────────────────────────────────────────────

    def _clear_project_nodes(self, project_id: str) -> None:
        """Clear all CodeNode data for a project."""
        try:
            self._conn.execute(
                "MATCH (n:CodeNode {project_id: $pid}) DETACH DELETE n",
                {"pid": project_id},
            )
        except Exception:
            pass

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Release database resources."""
        self._conn = None
        self._db = None
