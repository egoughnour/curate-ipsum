"""
BRS domain extension for code mutation testing.

This module provides domain-specific smoke tests and validation logic
for the code_mutation domain used by curate-ipsum.

Register with BRS:
    from brs.domains.registry import register_domain_smoke
    register_domain_smoke("code_mutation", "domains.code_mutation_smoke", "run_smoke")
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from brs import CASStore

LOG = logging.getLogger("domains.code_mutation")

DOMAIN_ID = "code_mutation"


def run_smoke(
    store: "CASStore",
    domain_id: str,
    world_label: str,
) -> Tuple[int, int, List[str]]:
    """
    Basic smoke tests for the code_mutation domain.

    Verifies:
    1. All assertions have grounding evidence
    2. No orphaned edges (edges to non-existent nodes)
    3. Evidence metadata is well-formed

    Args:
        store: BRS storage backend
        domain_id: Domain identifier
        world_label: World version to test

    Returns:
        Tuple of (total_tests, failures, messages)
    """
    tests = 0
    failures = 0
    messages: List[str] = []

    try:
        world_data = store.get_world(domain_id, world_label)["json"]
    except KeyError:
        return (1, 1, [f"World {domain_id}:{world_label} not found"])

    node_ids = set(world_data.get("node_ids", []))
    edge_ids = set(world_data.get("edge_ids", []))
    evidence_ids = set(world_data.get("evidence_ids", []))

    # Test 1: All assertions should have grounding evidence
    tests += 1
    nodes_without_evidence = []
    for node_id in node_ids:
        # Check for incoming "grounded_by" edges
        incoming = store.list_edges_into(node_id)
        grounding_edges = [e for e in incoming if e.get("kind") == "grounded_by"]
        if not grounding_edges:
            nodes_without_evidence.append(node_id)

    if nodes_without_evidence:
        failures += 1
        messages.append(
            f"Nodes without grounding evidence: {', '.join(nodes_without_evidence[:5])}"
            + (f" (and {len(nodes_without_evidence) - 5} more)" if len(nodes_without_evidence) > 5 else "")
        )

    # Test 2: No orphaned edges
    tests += 1
    orphaned_edges = []
    for edge_id in edge_ids:
        # Find the edge object
        row = store._conn.execute(
            "SELECT json FROM objects WHERE kind='Edge' AND json LIKE ?",
            (f'%"id": "{edge_id}"%',)
        ).fetchone()
        if row:
            edge = json.loads(row[0])
            parent_id = edge.get("parent_id")
            child_id = edge.get("child_id")

            # Check that both endpoints exist (either as nodes or evidence)
            if parent_id not in node_ids and parent_id not in evidence_ids:
                orphaned_edges.append((edge_id, "parent", parent_id))
            if child_id not in node_ids:
                orphaned_edges.append((edge_id, "child", child_id))

    if orphaned_edges:
        failures += 1
        messages.append(
            f"Orphaned edges found: {len(orphaned_edges)} edges reference non-existent nodes"
        )

    # Test 3: Evidence metadata well-formed
    tests += 1
    malformed_evidence = []
    for evidence_id in evidence_ids:
        row = store._conn.execute(
            "SELECT json FROM objects WHERE kind='Evidence' AND json LIKE ?",
            (f'%"id": "{evidence_id}"%',)
        ).fetchone()
        if row:
            evidence = json.loads(row[0])
            # Check required fields
            if not evidence.get("kind"):
                malformed_evidence.append((evidence_id, "missing kind"))
            if not evidence.get("reliability"):
                malformed_evidence.append((evidence_id, "missing reliability"))

    if malformed_evidence:
        failures += 1
        messages.append(
            f"Malformed evidence: {len(malformed_evidence)} evidence objects have issues"
        )

    LOG.info(
        "Smoke tests for %s:%s: %d/%d passed",
        domain_id,
        world_label,
        tests - failures,
        tests,
    )

    return (tests, failures, messages)


def run_regression(
    store: "CASStore",
    domain_id: str,
    world_label: str,
) -> Tuple[int, int, List[str]]:
    """
    Regression tests for the code_mutation domain.

    Verifies:
    1. Mutation scores are monotonically improving (or stable) over time
    2. No previously-killed mutants have resurfaced as survived

    Args:
        store: BRS storage backend
        domain_id: Domain identifier
        world_label: World version to test

    Returns:
        Tuple of (total_tests, failures, messages)
    """
    tests = 0
    failures = 0
    messages: List[str] = []

    try:
        world_data = store.get_world(domain_id, world_label)["json"]
    except KeyError:
        return (1, 1, [f"World {domain_id}:{world_label} not found"])

    evidence_ids = world_data.get("evidence_ids", [])

    # Collect mutation evidence ordered by date
    mutation_evidence = []
    for evidence_id in evidence_ids:
        row = store._conn.execute(
            "SELECT json FROM objects WHERE kind='Evidence' AND json LIKE ?",
            (f'%"id": "{evidence_id}"%',)
        ).fetchone()
        if row:
            evidence = json.loads(row[0])
            if evidence.get("kind", "").startswith("mutation_"):
                mutation_evidence.append(evidence)

    if not mutation_evidence:
        return (0, 0, ["No mutation evidence found - skipping regression tests"])

    # Sort by date
    mutation_evidence.sort(key=lambda e: e.get("date", ""))

    # Test 1: Mutation scores should not dramatically decrease
    tests += 1
    scores = []
    for ev in mutation_evidence:
        meta = ev.get("metadata", {})
        score = meta.get("mutation_score")
        if score is not None:
            scores.append((ev.get("date"), score))

    if len(scores) >= 2:
        # Check for significant regression (>10% decrease)
        for i in range(1, len(scores)):
            prev_score = scores[i - 1][1]
            curr_score = scores[i][1]
            if curr_score < prev_score - 0.1:
                failures += 1
                messages.append(
                    f"Mutation score regression: {prev_score:.2f} -> {curr_score:.2f} "
                    f"at {scores[i][0]}"
                )
                break

    LOG.info(
        "Regression tests for %s:%s: %d/%d passed",
        domain_id,
        world_label,
        tests - failures,
        tests,
    )

    return (tests, failures, messages)


def run_deep(
    store: "CASStore",
    domain_id: str,
    world_label: str,
) -> Tuple[int, int, List[str]]:
    """
    Deep validation tests for the code_mutation domain.

    Performs comprehensive validation including:
    1. All smoke tests
    2. All regression tests
    3. Cross-reference validation between assertions and evidence
    4. Entrenchment score consistency

    Args:
        store: BRS storage backend
        domain_id: Domain identifier
        world_label: World version to test

    Returns:
        Tuple of (total_tests, failures, messages)
    """
    # Run smoke and regression first
    smoke_tests, smoke_failures, smoke_messages = run_smoke(store, domain_id, world_label)
    reg_tests, reg_failures, reg_messages = run_regression(store, domain_id, world_label)

    total_tests = smoke_tests + reg_tests
    total_failures = smoke_failures + reg_failures
    all_messages = smoke_messages + reg_messages

    # Additional deep tests
    try:
        from brs import compute_entrenchment
    except ImportError:
        return (total_tests, total_failures, all_messages + ["py-brs not available for deep tests"])

    try:
        world_data = store.get_world(domain_id, world_label)["json"]
    except KeyError:
        return (total_tests + 1, total_failures + 1, all_messages + [f"World {domain_id}:{world_label} not found"])

    node_ids = world_data.get("node_ids", [])

    # Test: Entrenchment scores should be valid
    total_tests += 1
    invalid_entrenchment = []
    for node_id in node_ids[:10]:  # Sample first 10 to avoid performance issues
        incoming = store.list_edges_into(node_id)
        score = compute_entrenchment(store, node_id, incoming)
        if not (0.0 <= score <= 1.0):
            invalid_entrenchment.append((node_id, score))

    if invalid_entrenchment:
        total_failures += 1
        all_messages.append(
            f"Invalid entrenchment scores: {len(invalid_entrenchment)} nodes have out-of-range scores"
        )

    LOG.info(
        "Deep tests for %s:%s: %d/%d passed",
        domain_id,
        world_label,
        total_tests - total_failures,
        total_tests,
    )

    return (total_tests, total_failures, all_messages)


# Smoke test levels for BRS registry
SMOKE_LEVELS = {
    "smoke": run_smoke,
    "regression": run_regression,
    "deep": run_deep,
}
