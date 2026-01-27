"""
Theory Manager: Wraps py-brs operations with curate-ipsum-specific logic.

The TheoryManager provides a high-level interface for managing synthesis theories,
handling belief revision operations, and integrating evidence from mutation testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from brs import CASStore, ContractionResult, Node, WorldBundle

LOG = logging.getLogger("theory.manager")

# Default domain for code mutation testing
DEFAULT_DOMAIN = "code_mutation"
DEFAULT_WORLD = "green"


class TheoryManager:
    """
    Manages belief revision operations for code synthesis.

    This class wraps py-brs CASStore with curate-ipsum-specific logic,
    providing a simplified interface for:
    - Adding assertions (typed beliefs about code)
    - Contracting assertions (removing beliefs via AGM contraction)
    - Revising assertions (incorporating new evidence)
    - Computing entrenchment (belief resilience scores)

    Example:
        manager = TheoryManager(project_path)
        node = manager.add_assertion(
            assertion_type="behavior",
            content="function handles null input",
            evidence_id="test_123",
            confidence=0.8
        )
        score = manager.get_entrenchment(node["id"])
    """

    def __init__(
        self,
        project_path: Path,
        domain: str = DEFAULT_DOMAIN,
        world_label: str = DEFAULT_WORLD,
    ):
        """
        Initialize the TheoryManager.

        Args:
            project_path: Path to project directory (will create .curate_ipsum subdir)
            domain: BRS domain identifier
            world_label: Initial world version label
        """
        self._project_path = project_path
        self._domain = domain
        self._world_label = world_label
        self._store: Optional["CASStore"] = None
        self._store_path = project_path / ".curate_ipsum" / "beliefs.db"

    @property
    def store(self) -> "CASStore":
        """Lazy-load the CASStore."""
        if self._store is None:
            try:
                from brs import CASStore
            except ImportError as e:
                raise ImportError(
                    "py-brs is required for belief revision. "
                    "Install with: pip install py-brs>=2.0.0"
                ) from e

            # Ensure directory exists
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._store = CASStore(self._store_path)

        return self._store

    @property
    def domain(self) -> str:
        """Current domain identifier."""
        return self._domain

    @property
    def world_label(self) -> str:
        """Current world version label."""
        return self._world_label

    def _ensure_world_exists(self) -> None:
        """Ensure the current world exists, creating if necessary."""
        try:
            self.store.get_world(self._domain, self._world_label)
        except KeyError:
            # World doesn't exist - create it
            self._create_initial_world()

    def _create_initial_world(self) -> str:
        """Create the initial empty world."""
        from brs import WorldBundle, content_hash, canonical_json
        import datetime

        world = {
            "domain_id": self._domain,
            "version_label": self._world_label,
            "node_ids": [],
            "edge_ids": [],
            "evidence_ids": [],
            "pattern_ids": [],
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "notes": "Initial world for curate-ipsum",
            "metadata": {},
        }

        h = content_hash(world)
        js = canonical_json(world)

        # Store via internal API
        self.store._conn.execute(
            "INSERT OR IGNORE INTO objects(hash, kind, json) VALUES(?,?,?)",
            (h, "WorldBundle", js)
        )
        self.store._conn.execute(
            "INSERT OR REPLACE INTO worlds(domain_id, version_label, hash, created_utc) VALUES(?,?,?,?)",
            (self._domain, self._world_label, h, world["created_utc"])
        )
        self.store._conn.commit()

        LOG.info("Created initial world %s:%s", self._domain, self._world_label)
        return h

    def add_assertion(
        self,
        assertion_type: str,
        content: str,
        evidence_id: str,
        confidence: float = 0.5,
        region_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a new assertion (belief) to the theory with evidence grounding.

        Args:
            assertion_type: Type of assertion ("type", "behavior", "invariant", "contract")
            content: Human-readable description of the assertion
            evidence_id: ID of the evidence that grounds this assertion
            confidence: Confidence level (0.0 to 1.0)
            region_id: Optional code region this assertion relates to
            metadata: Optional additional metadata

        Returns:
            The created node as a dict

        Raises:
            ValueError: If assertion_type is invalid or confidence out of range
        """
        from brs import content_hash, canonical_json
        import datetime

        valid_types = {"type", "behavior", "invariant", "contract", "postcondition", "precondition"}
        if assertion_type not in valid_types:
            raise ValueError(f"Invalid assertion_type: {assertion_type}. Must be one of {valid_types}")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        self._ensure_world_exists()

        # Create node
        node_id = f"{assertion_type}_{content_hash({'type': assertion_type, 'content': content})[:12]}"
        node = {
            "id": node_id,
            "domain_id": self._domain,
            "kind": "Assertion",
            "properties": {
                "assertion_type": assertion_type,
                "content": content,
                "confidence": confidence,
                "region_id": region_id,
                **(metadata or {}),
            },
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }

        # Create grounding edge
        edge_id = f"grounds_{evidence_id}_{node_id}"
        edge = {
            "id": edge_id,
            "parent_id": evidence_id,
            "child_id": node_id,
            "kind": "grounded_by",
            "tier": 3,  # Default tier
            "confidence": confidence,
            "metadata": {},
        }

        # Store node
        node_hash = content_hash(node)
        self.store._conn.execute(
            "INSERT OR IGNORE INTO objects(hash, kind, json) VALUES(?,?,?)",
            (node_hash, "Node", canonical_json(node))
        )

        # Store edge
        edge_hash = content_hash(edge)
        self.store._conn.execute(
            "INSERT OR IGNORE INTO objects(hash, kind, json) VALUES(?,?,?)",
            (edge_hash, "Edge", canonical_json(edge))
        )

        # Update world to include new node and edge
        world_data = self.store.get_world(self._domain, self._world_label)["json"]
        new_node_ids = list(world_data.get("node_ids", [])) + [node_id]
        new_edge_ids = list(world_data.get("edge_ids", [])) + [edge_id]

        new_world = {
            **world_data,
            "node_ids": new_node_ids,
            "edge_ids": new_edge_ids,
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "notes": f"Added assertion {node_id}",
        }

        new_hash = content_hash(new_world)
        self.store._conn.execute(
            "INSERT OR IGNORE INTO objects(hash, kind, json) VALUES(?,?,?)",
            (new_hash, "WorldBundle", canonical_json(new_world))
        )
        self.store._conn.execute(
            "INSERT OR REPLACE INTO worlds(domain_id, version_label, hash, created_utc) VALUES(?,?,?,?)",
            (self._domain, self._world_label, new_hash, new_world["created_utc"])
        )
        self.store._conn.commit()

        LOG.info("Added assertion %s to %s:%s", node_id, self._domain, self._world_label)
        return node

    def contract_assertion(
        self,
        node_id: str,
        strategy: str = "entrenchment",
        cascade: bool = True,
    ) -> "ContractionResult":
        """
        Remove an assertion using AGM contraction.

        Args:
            node_id: ID of the node to contract (remove)
            strategy: Contraction strategy
                - "entrenchment": Remove target and dependents with lower entrenchment
                - "minimal": Remove only target and direct edges
                - "full_cascade": Remove target and all descendants
            cascade: Whether to cascade removal to unsupported dependents

        Returns:
            ContractionResult with details of what was removed

        Raises:
            KeyError: If node not found
            ValueError: If trying to contract a root node
        """
        from brs import contract

        self._ensure_world_exists()

        result = contract(
            self.store,
            self._domain,
            self._world_label,
            target_node_id=node_id,
            to_world=self._world_label,  # Contract in place
            strategy=strategy,
            cascade=cascade,
        )

        LOG.info(
            "Contracted %s from %s:%s - removed %d nodes, %d edges",
            node_id,
            self._domain,
            self._world_label,
            len(result.nodes_removed),
            len(result.edges_removed),
        )

        return result

    def revise_with_assertion(
        self,
        assertion_type: str,
        content: str,
        evidence_id: str,
        confidence: float = 0.5,
        contraction_strategy: str = "entrenchment",
    ) -> Tuple[str, Optional["ContractionResult"]]:
        """
        Revise theory by incorporating a new assertion.

        Implements AGM revision via the Levi identity:
        K*φ = (K÷¬φ)+φ

        If the new assertion contradicts existing beliefs,
        those are first contracted before adding the new assertion.

        Args:
            assertion_type: Type of assertion
            content: Assertion content
            evidence_id: Grounding evidence ID
            confidence: Confidence level
            contraction_strategy: Strategy for contracting contradictions

        Returns:
            Tuple of (new_world_hash, contraction_result_if_any)
        """
        from brs import revise, content_hash
        import datetime

        self._ensure_world_exists()

        # Create assertion node
        node_id = f"{assertion_type}_{content_hash({'type': assertion_type, 'content': content})[:12]}"
        assertion_node = {
            "id": node_id,
            "domain_id": self._domain,
            "kind": "Assertion",
            "properties": {
                "assertion_type": assertion_type,
                "content": content,
                "confidence": confidence,
            },
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }

        new_hash, contraction = revise(
            self.store,
            self._domain,
            self._world_label,
            assertion_node=assertion_node,
            to_world=self._world_label,  # Revise in place
            contraction_strategy=contraction_strategy,
        )

        LOG.info(
            "Revised %s:%s with %s (contraction=%s)",
            self._domain,
            self._world_label,
            node_id,
            contraction is not None,
        )

        return new_hash, contraction

    def get_entrenchment(self, node_id: str) -> float:
        """
        Get the entrenchment score for a node.

        Entrenchment measures how resilient a belief is to removal.
        Higher scores (closer to 1.0) mean the belief is more entrenched.

        Args:
            node_id: ID of the node to score

        Returns:
            Entrenchment score (0.0 to 1.0)
        """
        from brs import compute_entrenchment

        self._ensure_world_exists()

        # Get incoming edges to this node
        incoming_edges = self.store.list_edges_into(node_id)

        return compute_entrenchment(self.store, node_id, incoming_edges)

    def list_assertions(
        self,
        assertion_type: Optional[str] = None,
        region_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all assertions in the current world.

        Args:
            assertion_type: Filter by assertion type (optional)
            region_id: Filter by region ID (optional)

        Returns:
            List of assertion nodes
        """
        import json

        self._ensure_world_exists()

        world_data = self.store.get_world(self._domain, self._world_label)["json"]
        node_ids = world_data.get("node_ids", [])

        assertions = []
        for node_id in node_ids:
            # Query for node
            row = self.store._conn.execute(
                "SELECT json FROM objects WHERE kind='Node' AND json LIKE ?",
                (f'%"id": "{node_id}"%',)
            ).fetchone()

            if row:
                node = json.loads(row[0])
                props = node.get("properties", {})

                # Apply filters
                if assertion_type and props.get("assertion_type") != assertion_type:
                    continue
                if region_id and props.get("region_id") != region_id:
                    continue

                assertions.append(node)

        return assertions

    def get_theory_snapshot(self) -> Dict[str, Any]:
        """
        Get the current theory state as a snapshot.

        Returns:
            WorldBundle data for the current world
        """
        self._ensure_world_exists()
        return self.store.get_world(self._domain, self._world_label)["json"]

    def store_evidence(self, evidence: "Any") -> str:
        """
        Store a BRS Evidence object.

        Args:
            evidence: Evidence object from adapters

        Returns:
            Evidence ID
        """
        from brs import content_hash, canonical_json
        import datetime

        evidence_dict = {
            "id": evidence.id,
            "citation": evidence.citation,
            "kind": evidence.kind,
            "reliability": evidence.reliability,
            "date": evidence.date,
            "metadata": evidence.metadata,
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }

        h = content_hash(evidence_dict)
        self.store._conn.execute(
            "INSERT OR IGNORE INTO objects(hash, kind, json) VALUES(?,?,?)",
            (h, "Evidence", canonical_json(evidence_dict))
        )

        # Update world to include evidence
        self._ensure_world_exists()
        world_data = self.store.get_world(self._domain, self._world_label)["json"]
        new_evidence_ids = list(world_data.get("evidence_ids", [])) + [evidence.id]

        new_world = {
            **world_data,
            "evidence_ids": new_evidence_ids,
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }

        new_hash = content_hash(new_world)
        self.store._conn.execute(
            "INSERT OR IGNORE INTO objects(hash, kind, json) VALUES(?,?,?)",
            (new_hash, "WorldBundle", canonical_json(new_world))
        )
        self.store._conn.execute(
            "INSERT OR REPLACE INTO worlds(domain_id, version_label, hash, created_utc) VALUES(?,?,?,?)",
            (self._domain, self._world_label, new_hash, new_world["created_utc"])
        )
        self.store._conn.commit()

        LOG.info("Stored evidence %s", evidence.id)
        return evidence.id
