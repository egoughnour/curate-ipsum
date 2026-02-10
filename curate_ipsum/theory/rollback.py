"""
Rollback mechanism for theory state recovery.

Provides an explicit API for reverting to prior world states, building
on CASStore's content-addressed world versioning and the provenance DAG's
event history.

Design: rollback changes the world_label pointer to reference a prior
world_hash — it does NOT copy or mutate any worlds (content-addressable
storage means all historical worlds are already preserved).
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from curate_ipsum.theory.provenance import ProvenanceDAG, RevisionEvent, RevisionType

if TYPE_CHECKING:
    from curate_ipsum.theory.manager import TheoryManager

LOG = logging.getLogger("theory.rollback")


class RollbackError(Exception):
    """Raised when a rollback operation fails."""

    pass


@dataclass
class Checkpoint:
    """Named bookmark for a world state."""

    name: str
    world_hash: str
    timestamp: str
    reason: str = ""


class RollbackManager:
    """
    Manages rollback operations on the synthesis theory.

    Provides methods to revert to prior world states, undo operations,
    and create/restore named checkpoints.
    """

    def __init__(self, manager: "TheoryManager", dag: ProvenanceDAG) -> None:
        """
        Initialize the RollbackManager.

        Args:
            manager: The TheoryManager to operate on
            dag: The provenance DAG for event history
        """
        self._manager = manager
        self._dag = dag
        self._checkpoints: list[Checkpoint] = []

    def rollback_to(self, target_world_hash: str) -> None:
        """
        Revert the theory to a prior world state.

        This changes the world_label to point to the target world hash.
        Does NOT modify any existing world data (content-addressable).

        Args:
            target_world_hash: Hash of the world to revert to

        Raises:
            RollbackError: If the target world doesn't exist
        """
        # Verify the target world exists in the store
        try:
            self._manager.store._conn.execute(
                "SELECT hash FROM objects WHERE hash=? AND kind='WorldBundle'",
                (target_world_hash,),
            ).fetchone()
        except Exception:
            pass

        row = self._manager.store._conn.execute(
            "SELECT hash FROM objects WHERE hash=? AND kind='WorldBundle'",
            (target_world_hash,),
        ).fetchone()

        if not row:
            raise RollbackError(f"Target world {target_world_hash} not found in store")

        # Get current world hash before rollback
        _current = self._manager.get_theory_snapshot()
        from_hash = None
        try:
            world_row = self._manager.store._conn.execute(
                "SELECT hash FROM worlds WHERE domain_id=? AND version_label=?",
                (self._manager.domain, self._manager.world_label),
            ).fetchone()
            if world_row:
                from_hash = world_row[0]
        except Exception:
            pass

        # Update the world pointer
        now = datetime.datetime.utcnow().isoformat() + "Z"
        self._manager.store._conn.execute(
            "INSERT OR REPLACE INTO worlds(domain_id, version_label, hash, created_utc) VALUES(?,?,?,?)",
            (
                self._manager.domain,
                self._manager.world_label,
                target_world_hash,
                now,
            ),
        )
        self._manager.store._conn.commit()

        # Record rollback event in provenance
        event = RevisionEvent(
            event_type=RevisionType.ROLLBACK,
            timestamp=now,
            from_world_hash=from_hash,
            to_world_hash=target_world_hash,
            reason=f"Rolled back to world {target_world_hash[:12]}",
        )
        self._dag.add_event(event)

        LOG.info(
            "Rolled back %s:%s to world %s",
            self._manager.domain,
            self._manager.world_label,
            target_world_hash[:12],
        )

    def undo_last(self, n: int = 1) -> list[RevisionEvent]:
        """
        Undo the last N revision operations.

        Walks backward through the provenance DAG to find the world
        state before the last N events, then rolls back to it.

        Args:
            n: Number of operations to undo (default: 1)

        Returns:
            List of undone events

        Raises:
            RollbackError: If there aren't enough events to undo
        """
        events = self._dag.get_history()
        if len(events) < n:
            raise RollbackError(f"Cannot undo {n} operations: only {len(events)} events in history")

        # Find the world state before the Nth-from-last event
        target_event = events[-(n)]
        target_hash = target_event.from_world_hash

        if target_hash is None:
            raise RollbackError(f"Event {target_event.event_type.value} has no from_world_hash")

        undone_events = events[-n:]
        self.rollback_to(target_hash)

        LOG.info("Undid %d operations, rolled back to %s", n, target_hash[:12])
        return undone_events

    def create_checkpoint(self, name: str, reason: str = "") -> Checkpoint:
        """
        Create a named checkpoint of the current world state.

        Args:
            name: Human-readable checkpoint name
            reason: Why this checkpoint was created

        Returns:
            The created Checkpoint
        """
        # Get current world hash
        world_row = self._manager.store._conn.execute(
            "SELECT hash FROM worlds WHERE domain_id=? AND version_label=?",
            (self._manager.domain, self._manager.world_label),
        ).fetchone()

        if not world_row:
            raise RollbackError("No current world found to checkpoint")

        checkpoint = Checkpoint(
            name=name,
            world_hash=world_row[0],
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            reason=reason,
        )
        self._checkpoints.append(checkpoint)

        LOG.info("Created checkpoint '%s' at world %s", name, checkpoint.world_hash[:12])
        return checkpoint

    def list_checkpoints(self) -> list[Checkpoint]:
        """Get all named checkpoints."""
        return list(self._checkpoints)

    def restore_checkpoint(self, name: str) -> None:
        """
        Restore a named checkpoint.

        Args:
            name: The checkpoint name to restore

        Raises:
            RollbackError: If checkpoint not found
        """
        checkpoint = next((c for c in self._checkpoints if c.name == name), None)
        if checkpoint is None:
            raise RollbackError(f"Checkpoint '{name}' not found")

        self.rollback_to(checkpoint.world_hash)
        LOG.info("Restored checkpoint '%s'", name)

    def list_world_history(self) -> list[tuple[str, str, str]]:
        """
        List all historical world states with timestamps.

        Returns:
            List of (world_hash, timestamp, reason) tuples
        """
        result = []
        for event in self._dag.get_history():
            if event.to_world_hash:
                result.append(
                    (
                        event.to_world_hash,
                        event.timestamp,
                        event.reason or event.event_type.value,
                    )
                )
        return result
