"""
Provenance DAG: tracks causal chains of belief evolution.

The provenance DAG is an append-only log of revision events that records
WHY each world state exists. While CASStore preserves WHAT each world
looks like (via content-addressed snapshots), the provenance DAG captures
the causal reasoning: which evidence triggered which revision, which
assertions were added or removed, and why.

Design decisions:
- Append-only: events are immutable once recorded (audit trail)
- DAG structure derived from event ordering (no explicit parent pointers)
- Serialized to CASStore as a single object for persistence
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from brs import CASStore

LOG = logging.getLogger("theory.provenance")


class RevisionType(StrEnum):
    """Type of belief revision operation."""

    EXPAND = "expand"  # New assertion added
    CONTRACT = "contract"  # Assertion removed
    REVISE = "revise"  # Contradiction resolved + new assertion added
    EVIDENCE = "evidence"  # New evidence stored (no belief change)
    ROLLBACK = "rollback"  # Reverted to prior state


@dataclass
class RevisionEvent:
    """
    A single event in the theory's revision history.

    Records what happened, why, and the before/after world state hashes.
    """

    event_type: RevisionType
    timestamp: str  # ISO 8601 UTC
    assertion_id: str | None = None  # Which assertion was affected
    evidence_id: str | None = None  # Which evidence triggered this
    from_world_hash: str | None = None  # World state before
    to_world_hash: str | None = None  # World state after
    strategy: str | None = None  # For contractions: entrenchment/minimal/full_cascade
    reason: str | None = None  # Human-readable explanation
    nodes_removed: list[str] = field(default_factory=list)
    nodes_added: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "assertion_id": self.assertion_id,
            "evidence_id": self.evidence_id,
            "from_world_hash": self.from_world_hash,
            "to_world_hash": self.to_world_hash,
            "strategy": self.strategy,
            "reason": self.reason,
            "nodes_removed": self.nodes_removed,
            "nodes_added": self.nodes_added,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RevisionEvent":
        """Deserialize from dict."""
        return cls(
            event_type=RevisionType(data["event_type"]),
            timestamp=data["timestamp"],
            assertion_id=data.get("assertion_id"),
            evidence_id=data.get("evidence_id"),
            from_world_hash=data.get("from_world_hash"),
            to_world_hash=data.get("to_world_hash"),
            strategy=data.get("strategy"),
            reason=data.get("reason"),
            nodes_removed=data.get("nodes_removed", []),
            nodes_added=data.get("nodes_added", []),
            metadata=data.get("metadata", {}),
        )


class ProvenanceDAG:
    """
    Directed acyclic graph of theory revision events.

    Stores an append-only log of all revision operations and provides
    query methods for understanding belief evolution.
    """

    def __init__(self) -> None:
        self._events: list[RevisionEvent] = []
        # Indexes for fast lookups
        self._by_assertion: dict[str, list[int]] = {}  # assertion_id → event indices
        self._by_world: dict[str, int] = {}  # to_world_hash → event index

    @property
    def events(self) -> list[RevisionEvent]:
        """All events in chronological order."""
        return list(self._events)

    def add_event(self, event: RevisionEvent) -> None:
        """
        Record a new revision event.

        Args:
            event: The revision event to record
        """
        idx = len(self._events)
        self._events.append(event)

        # Index by assertion
        if event.assertion_id:
            self._by_assertion.setdefault(event.assertion_id, []).append(idx)

        # Index by target world
        if event.to_world_hash:
            self._by_world[event.to_world_hash] = idx

        for node_id in event.nodes_added:
            self._by_assertion.setdefault(node_id, []).append(idx)

        LOG.debug(
            "Recorded %s event: assertion=%s, world=%s→%s",
            event.event_type.value,
            event.assertion_id,
            event.from_world_hash and event.from_world_hash[:8],
            event.to_world_hash and event.to_world_hash[:8],
        )

    def get_history(self) -> list[RevisionEvent]:
        """Get all events in chronological order."""
        return list(self._events)

    def get_path(self, from_hash: str, to_hash: str) -> list[RevisionEvent]:
        """
        Get the chain of events between two world states.

        Args:
            from_hash: Starting world hash
            to_hash: Ending world hash

        Returns:
            List of events connecting the two states (may be empty)
        """
        # Build world→world chain
        path_events = []
        current = to_hash

        visited = set()
        while current and current != from_hash and current not in visited:
            visited.add(current)
            idx = self._by_world.get(current)
            if idx is None:
                break
            event = self._events[idx]
            path_events.append(event)
            current = event.from_world_hash

        if current == from_hash:
            path_events.reverse()
            return path_events

        return []  # No path found

    def why_believe(self, assertion_id: str) -> list[str]:
        """
        Trace which evidence grounds an assertion.

        Args:
            assertion_id: The assertion to trace

        Returns:
            List of evidence IDs that support this assertion
        """
        evidence_ids = []
        indices = self._by_assertion.get(assertion_id, [])

        for idx in indices:
            event = self._events[idx]
            if event.event_type in (RevisionType.EXPAND, RevisionType.REVISE):
                if event.evidence_id:
                    evidence_ids.append(event.evidence_id)

        return evidence_ids

    def when_added(self, assertion_id: str) -> RevisionEvent | None:
        """
        Find the first event that added this assertion.

        Args:
            assertion_id: The assertion to look up

        Returns:
            The expansion event, or None if not found
        """
        indices = self._by_assertion.get(assertion_id, [])

        for idx in indices:
            event = self._events[idx]
            if event.event_type in (RevisionType.EXPAND, RevisionType.REVISE):
                if event.assertion_id == assertion_id or assertion_id in event.nodes_added:
                    return event

        return None

    def when_removed(self, assertion_id: str) -> RevisionEvent | None:
        """
        Find the event that removed this assertion.

        Args:
            assertion_id: The assertion to look up

        Returns:
            The contraction/revise event, or None if still present
        """
        indices = self._by_assertion.get(assertion_id, [])

        for idx in indices:
            event = self._events[idx]
            if event.event_type in (RevisionType.CONTRACT, RevisionType.REVISE):
                if assertion_id in event.nodes_removed:
                    return event

        return None

    def belief_stability(self, assertion_id: str) -> float:
        """
        Measure how stable an assertion is.

        Returns a score from 0.0 (constantly revised) to 1.0 (never touched).
        Formula: 1.0 - (revisions / (revisions + 1))
        An assertion that's been added once and never revised = 1.0.
        An assertion added, removed, re-added = 0.5.

        Args:
            assertion_id: The assertion to measure

        Returns:
            Stability score (0.0 to 1.0)
        """
        indices = self._by_assertion.get(assertion_id, [])
        if not indices:
            return 1.0  # Unknown = assumed stable

        revision_count = 0
        for idx in indices:
            event = self._events[idx]
            if event.event_type in (
                RevisionType.CONTRACT,
                RevisionType.REVISE,
                RevisionType.ROLLBACK,
            ):
                # Count removals and revisions
                if assertion_id in event.nodes_removed or (
                    event.event_type == RevisionType.REVISE and event.assertion_id == assertion_id
                ):
                    revision_count += 1

        # stability = 1 / (1 + revisions)
        return 1.0 / (1.0 + revision_count)

    def get_world_hashes(self) -> list[str]:
        """Get all world hashes in chronological order."""
        hashes = []
        seen = set()
        for event in self._events:
            if event.from_world_hash and event.from_world_hash not in seen:
                hashes.append(event.from_world_hash)
                seen.add(event.from_world_hash)
            if event.to_world_hash and event.to_world_hash not in seen:
                hashes.append(event.to_world_hash)
                seen.add(event.to_world_hash)
        return hashes

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire DAG to a dict."""
        return {
            "events": [e.to_dict() for e in self._events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceDAG":
        """Deserialize from a dict."""
        dag = cls()
        for event_data in data.get("events", []):
            dag.add_event(RevisionEvent.from_dict(event_data))
        return dag


class ProvenanceStore:
    """
    Persistence layer for ProvenanceDAG using CASStore.

    The provenance DAG is stored as a single CASStore object with kind
    'ProvenanceDAG', keyed by domain_id.
    """

    PROVENANCE_KEY_PREFIX = "provenance_"

    @staticmethod
    def save(store: "CASStore", domain_id: str, dag: ProvenanceDAG) -> str:
        """
        Save the provenance DAG to CASStore.

        Args:
            store: The CASStore instance
            domain_id: Domain identifier
            dag: The provenance DAG to save

        Returns:
            Content hash of the stored DAG
        """
        from brs import canonical_json, content_hash

        dag_data = {
            "domain_id": domain_id,
            "kind": "ProvenanceDAG",
            "event_count": len(dag.events),
            **dag.to_dict(),
        }

        h = content_hash(dag_data)
        js = canonical_json(dag_data)

        store._conn.execute(
            "INSERT OR REPLACE INTO objects(hash, kind, json) VALUES(?,?,?)",
            (h, "ProvenanceDAG", js),
        )

        # Store a reference so we can find it by domain
        key = f"{ProvenanceStore.PROVENANCE_KEY_PREFIX}{domain_id}"
        store._conn.execute(
            "INSERT OR REPLACE INTO objects(hash, kind, json) VALUES(?,?,?)",
            (key, "ProvenanceRef", json.dumps({"hash": h, "domain_id": domain_id})),
        )
        store._conn.commit()

        LOG.debug("Saved provenance DAG for %s (%d events)", domain_id, len(dag.events))
        return h

    @staticmethod
    def load(store: "CASStore", domain_id: str) -> ProvenanceDAG:
        """
        Load the provenance DAG from CASStore.

        Args:
            store: The CASStore instance
            domain_id: Domain identifier

        Returns:
            The provenance DAG (empty if none found)
        """
        key = f"{ProvenanceStore.PROVENANCE_KEY_PREFIX}{domain_id}"

        row = store._conn.execute(
            "SELECT json FROM objects WHERE hash=? AND kind='ProvenanceRef'",
            (key,),
        ).fetchone()

        if not row:
            LOG.debug("No provenance DAG found for %s, creating empty", domain_id)
            return ProvenanceDAG()

        ref = json.loads(row[0])
        dag_hash = ref.get("hash")

        dag_row = store._conn.execute(
            "SELECT json FROM objects WHERE hash=? AND kind='ProvenanceDAG'",
            (dag_hash,),
        ).fetchone()

        if not dag_row:
            LOG.warning("Provenance DAG hash %s not found, creating empty", dag_hash)
            return ProvenanceDAG()

        dag_data = json.loads(dag_row[0])
        dag = ProvenanceDAG.from_dict(dag_data)

        LOG.debug("Loaded provenance DAG for %s (%d events)", domain_id, len(dag.events))
        return dag
