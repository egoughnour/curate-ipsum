"""
Theory Manager: Wraps py-brs operations with curate-ipsum-specific logic.

The TheoryManager provides a high-level interface for managing synthesis theories,
handling belief revision operations, and integrating evidence from mutation testing.

M3 additions: provenance tracking, rollback, failure analysis, typed assertions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from theory.assertions import (
    Assertion,
    AssertionKind,
    ContradictionDetector,
    assertion_to_node_dict,
    node_dict_to_assertion,
)
from theory.failure_analyzer import FailureAnalysis, FailureModeAnalyzer
from theory.provenance import ProvenanceDAG, ProvenanceStore, RevisionEvent, RevisionType
from theory.rollback import RollbackManager

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
        self._provenance_dag: Optional[ProvenanceDAG] = None
        self._rollback_manager: Optional[RollbackManager] = None

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

    @property
    def provenance_dag(self) -> ProvenanceDAG:
        """Lazy-load the provenance DAG from CASStore."""
        if self._provenance_dag is None:
            try:
                self._provenance_dag = ProvenanceStore.load(
                    self.store, self._domain
                )
            except Exception:
                self._provenance_dag = ProvenanceDAG()
        return self._provenance_dag

    def _save_provenance(self) -> None:
        """Persist the provenance DAG to CASStore."""
        try:
            ProvenanceStore.save(self.store, self._domain, self.provenance_dag)
        except Exception as exc:
            LOG.warning("Failed to save provenance DAG: %s", exc)

    def _get_current_world_hash(self) -> Optional[str]:
        """Get the hash of the current world state."""
        try:
            row = self.store._conn.execute(
                "SELECT hash FROM worlds WHERE domain_id=? AND version_label=?",
                (self._domain, self._world_label),
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def get_rollback_manager(self) -> RollbackManager:
        """Get or create a RollbackManager."""
        if self._rollback_manager is None:
            self._rollback_manager = RollbackManager(self, self.provenance_dag)
        return self._rollback_manager

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

        # Record provenance event
        from_hash = content_hash(world_data) if world_data else None
        event = RevisionEvent(
            event_type=RevisionType.EXPAND,
            timestamp=node["created_utc"],
            assertion_id=node_id,
            evidence_id=evidence_id,
            from_world_hash=from_hash,
            to_world_hash=new_hash,
            reason=f"Added {assertion_type} assertion: {content[:60]}",
            nodes_added=[node_id],
        )
        self.provenance_dag.add_event(event)
        self._save_provenance()

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
        import datetime

        self._ensure_world_exists()
        from_hash = self._get_current_world_hash()

        result = contract(
            self.store,
            self._domain,
            self._world_label,
            target_node_id=node_id,
            to_world=self._world_label,  # Contract in place
            strategy=strategy,
            cascade=cascade,
        )

        to_hash = self._get_current_world_hash()

        # Record provenance event
        event = RevisionEvent(
            event_type=RevisionType.CONTRACT,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            assertion_id=node_id,
            from_world_hash=from_hash,
            to_world_hash=to_hash,
            strategy=strategy,
            reason=f"Contracted {node_id} via {strategy}",
            nodes_removed=list(result.nodes_removed),
        )
        self.provenance_dag.add_event(event)
        self._save_provenance()

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
        from_hash = self._get_current_world_hash()

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

        # Record provenance event
        nodes_removed = list(contraction.nodes_removed) if contraction else []
        event = RevisionEvent(
            event_type=RevisionType.REVISE,
            timestamp=assertion_node["created_utc"],
            assertion_id=node_id,
            evidence_id=evidence_id,
            from_world_hash=from_hash,
            to_world_hash=new_hash,
            strategy=contraction_strategy,
            reason=f"Revised with {assertion_type}: {content[:60]}",
            nodes_removed=nodes_removed,
            nodes_added=[node_id],
        )
        self.provenance_dag.add_event(event)
        self._save_provenance()

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
            # Query for node — use simple substring match (canonical_json may
            # omit spaces after colons)
            row = self.store._conn.execute(
                "SELECT json FROM objects WHERE kind='Node' AND json LIKE ?",
                (f'%"id":"{node_id}"%',)
            ).fetchone()

            # Fallback: try with space after colon (non-canonical JSON)
            if row is None:
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

        from_hash = content_hash(world_data)

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

        # Record provenance event for evidence storage
        event = RevisionEvent(
            event_type=RevisionType.EVIDENCE,
            timestamp=evidence_dict["created_utc"],
            evidence_id=evidence.id,
            from_world_hash=from_hash,
            to_world_hash=new_hash,
            reason=f"Stored evidence {evidence.id}",
        )
        self.provenance_dag.add_event(event)
        self._save_provenance()

        LOG.info("Stored evidence %s", evidence.id)
        return evidence.id

    # =========================================================================
    # M3: Provenance query methods (delegate to ProvenanceDAG)
    # =========================================================================

    def why_believe(self, assertion_id: str) -> List[str]:
        """
        Trace which evidence grounds an assertion.

        Args:
            assertion_id: The assertion to trace

        Returns:
            List of evidence IDs that support this assertion
        """
        return self.provenance_dag.why_believe(assertion_id)

    def when_added(self, assertion_id: str) -> Optional[RevisionEvent]:
        """
        Find when an assertion was first added.

        Args:
            assertion_id: The assertion to look up

        Returns:
            The expansion event, or None
        """
        return self.provenance_dag.when_added(assertion_id)

    def belief_stability(self, assertion_id: str) -> float:
        """
        Measure how stable an assertion is.

        Returns 1.0 for never-revised beliefs, lower for frequently revised.

        Args:
            assertion_id: The assertion to measure

        Returns:
            Stability score (0.0 to 1.0)
        """
        return self.provenance_dag.belief_stability(assertion_id)

    def get_provenance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the provenance DAG.

        Returns:
            Dict with event counts, world hashes, and recent events
        """
        dag = self.provenance_dag
        events = dag.get_history()

        type_counts: Dict[str, int] = {}
        for e in events:
            t = e.event_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        recent = events[-10:] if len(events) > 10 else events

        return {
            "total_events": len(events),
            "event_type_counts": type_counts,
            "world_hashes": dag.get_world_hashes(),
            "recent_events": [e.to_dict() for e in recent],
        }

    # =========================================================================
    # M3: Failure analysis
    # =========================================================================

    def analyze_failure(
        self,
        error_message: str = "",
        test_pass_rate: Optional[float] = None,
        mutation_score: Optional[float] = None,
        failing_tests: Optional[List[str]] = None,
        region_id: Optional[str] = None,
    ) -> FailureAnalysis:
        """
        Analyze why a synthesis attempt failed.

        Combines error message classification, overfitting/underfitting
        detection, and assertion-based contraction suggestions.

        Args:
            error_message: Error output from test execution
            test_pass_rate: Fraction of tests passing (0.0 to 1.0)
            mutation_score: Fraction of mutants killed (0.0 to 1.0)
            failing_tests: Names of failing tests
            region_id: Region where the patch was applied

        Returns:
            FailureAnalysis with mode, confidence, and suggestions
        """
        # Get current assertions for contraction suggestions
        assertions = []
        try:
            assertion_dicts = self.list_assertions(region_id=region_id)
            for ad in assertion_dicts:
                try:
                    assertions.append(node_dict_to_assertion(ad))
                except (ValueError, KeyError):
                    pass
        except Exception:
            pass

        return FailureModeAnalyzer.analyze(
            error_message=error_message,
            test_pass_rate=test_pass_rate,
            mutation_score=mutation_score,
            failing_tests=failing_tests,
            assertions=assertions,
            region_id=region_id,
        )

    # =========================================================================
    # M3: Contradiction detection
    # =========================================================================

    def find_contradictions(
        self,
        assertion_type: str,
        content: str,
        confidence: float = 0.5,
        region_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find existing assertions that would contradict a new assertion.

        Args:
            assertion_type: Type of the new assertion
            content: Content of the new assertion
            confidence: Confidence of the new assertion
            region_id: Region of the new assertion

        Returns:
            List of contradicting assertion node dicts
        """
        new_assertion = Assertion(
            id="__candidate__",
            kind=AssertionKind(assertion_type),
            content=content,
            confidence=confidence,
            region_id=region_id,
        )

        existing = []
        try:
            assertion_dicts = self.list_assertions(region_id=region_id)
            for ad in assertion_dicts:
                try:
                    existing.append(node_dict_to_assertion(ad))
                except (ValueError, KeyError):
                    pass
        except Exception:
            pass

        contradictions = ContradictionDetector.find_contradictions(
            new_assertion, existing
        )

        # Return as dicts for MCP compatibility
        return [assertion_to_node_dict(c) for c in contradictions]
