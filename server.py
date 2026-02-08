from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools import (
    DATA_DIR,
    history_tool,
    region_metrics_tool,
    run_integration_tests,
    run_mutation_tests,
    run_unit_tests,
)

LOG_LEVEL = os.environ.get("MUTATION_TOOL_LOG_LEVEL", "INFO").upper()

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - dependency is optional at import time
    FastMCP = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_server() -> "FastMCP":
    if FastMCP is None:
        raise SystemExit(
            "The mcp package is required to run the server. "
            "Install it with `pip install mcp` or `pip install mcp[fastmcp]`."
        ) from _IMPORT_ERROR
    return FastMCP("mutation-tool-server")


def _validate_required(name: str, value: Optional[str]) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Missing required field: {name}")


def _json_payload(model) -> dict:
    return model.model_dump(mode="json")


def build_server() -> "FastMCP":
    server = _require_server()

    @server.tool(
            description="Run unit tests for a project and return a summarized result."
    )
    async def run_unit_tests_tool(
        projectId: str,
        commitSha: str,
        command: str,
        workingDirectory: str,
        regionId: Optional[str] = None,
        framework: str = "generic",
    ) -> dict:
        _validate_required("projectId", projectId)
        _validate_required("commitSha", commitSha)
        _validate_required("command", command)
        _validate_required("workingDirectory", workingDirectory)
        run = await run_unit_tests(
            projectId=projectId,
            commitSha=commitSha,
            command=command,
            workingDirectory=workingDirectory,
            regionId=regionId,
            framework=framework,
        )
        return _json_payload(run)

    @server.tool(
            description="Run integration tests for a project and return a summarized result."
    )
    async def run_integration_tests_tool(
        projectId: str,
        commitSha: str,
        command: str,
        workingDirectory: str,
        regionId: Optional[str] = None,
        framework: str = "generic",
    ) -> dict:
        _validate_required("projectId", projectId)
        _validate_required("commitSha", commitSha)
        _validate_required("command", command)
        _validate_required("workingDirectory", workingDirectory)
        run = await run_integration_tests(
            projectId=projectId,
            commitSha=commitSha,
            command=command,
            workingDirectory=workingDirectory,
            regionId=regionId,
            framework=framework,
        )
        return _json_payload(run)

    @server.tool(
        description=(
            "Run mutation tests and return summarized mutation statistics. "
            "Supports multiple frameworks: stryker (JS/TS), mutmut (Python). "
            "If tool is not specified, auto-detects based on project structure."
        )
    )
    async def run_mutation_tests_tool(
        projectId: str,
        commitSha: str,
        command: str,
        workingDirectory: str,
        regionId: Optional[str] = None,
        tool: Optional[str] = None,  # Now optional - auto-detected
        reportPath: Optional[str] = None,
    ) -> dict:
        _validate_required("projectId", projectId)
        _validate_required("commitSha", commitSha)
        _validate_required("command", command)
        _validate_required("workingDirectory", workingDirectory)
        run = await run_mutation_tests(
            projectId=projectId,
            commitSha=commitSha,
            command=command,
            workingDirectory=workingDirectory,
            regionId=regionId,
            tool=tool,
            reportPath=reportPath,
        )
        return _json_payload(run)

    @server.tool(
            description="Return recent unit, integration, and mutation runs for a project and optional region."
    )
    def get_run_history_tool(projectId: str, regionId: Optional[str] = None, limit: Optional[int] = None) -> dict:
        _validate_required("projectId", projectId)
        history = history_tool(projectId=projectId, regionId=regionId, limit=limit)
        return _json_payload(history)

    @server.tool(
            description="Compute PID-like metrics and mutation score for a specific region within a project."
                 )
    def get_region_metrics_tool(projectId: str, commitSha: str, regionId: str) -> dict:
        _validate_required("projectId", projectId)
        _validate_required("commitSha", commitSha)
        _validate_required("regionId", regionId)
        metrics = region_metrics_tool(projectId=projectId, commitSha=commitSha, regionId=regionId)
        return _json_payload(metrics)

    # =========================================================================
    # Framework Detection & Region Tools
    # =========================================================================

    @server.tool(
        description=(
            "Detect available mutation testing frameworks and project language. "
            "Returns detected frameworks with confidence scores and a recommendation "
            "for which framework to use."
        )
    )
    def detect_frameworks_tool(workingDirectory: str) -> dict:
        """Detect mutation frameworks in a project."""
        _validate_required("workingDirectory", workingDirectory)

        from parsers import (
            detect_available_frameworks,
            detect_language,
            recommend_framework,
        )

        language = detect_language(workingDirectory)
        frameworks = detect_available_frameworks(workingDirectory)
        recommendation = recommend_framework(workingDirectory)

        return {
            "language": {
                "primary": language.primary,
                "secondary": language.secondary,
                "confidence": round(language.confidence, 3),
            },
            "detected_frameworks": [
                {
                    "framework": f.framework.value,
                    "confidence": round(f.confidence, 3),
                    "evidence": f.evidence,
                }
                for f in frameworks
            ],
            "recommendation": {
                "framework": recommendation.framework.value,
                "confidence": round(recommendation.confidence, 3),
                "evidence": recommendation.evidence,
            },
        }

    @server.tool(
        description=(
            "Parse a region identifier string into its components. "
            "Regions use a hierarchical format: file:<path>::class:<name>::func:<name>::lines:<start>-<end>. "
            "Useful for understanding region hierarchy."
        )
    )
    def parse_region_tool(regionId: str) -> dict:
        """Parse a region string into components."""
        _validate_required("regionId", regionId)

        from regions import Region

        region = Region.from_string(regionId)

        return {
            "regionId": region.to_string(),
            "level": region.level.value,
            "file_path": region.file_path,
            "class_name": region.class_name,
            "func_name": region.func_name,
            "line_start": region.line_start,
            "line_end": region.line_end,
        }

    @server.tool(
        description=(
            "Check if one region contains or overlaps another. "
            "Useful for aggregating metrics across related regions. "
            "A file region contains all functions within it; a function region contains its line ranges."
        )
    )
    def check_region_relationship_tool(
        regionA: str,
        regionB: str,
    ) -> dict:
        """Check containment/overlap relationship between regions."""
        _validate_required("regionA", regionA)
        _validate_required("regionB", regionB)

        from regions import Region

        a = Region.from_string(regionA)
        b = Region.from_string(regionB)

        return {
            "a": a.to_string(),
            "b": b.to_string(),
            "a_contains_b": a.contains(b),
            "b_contains_a": b.contains(a),
            "overlaps": a.overlaps(b),
        }

    @server.tool(
        description=(
            "Create a region identifier for a specific code location. "
            "Use level='file' for whole file, 'function' for a function, "
            "'class' for a class, or 'lines' for a line range."
        )
    )
    def create_region_tool(
        filePath: str,
        level: str = "file",
        className: Optional[str] = None,
        funcName: Optional[str] = None,
        lineStart: Optional[int] = None,
        lineEnd: Optional[int] = None,
    ) -> dict:
        """Create a region identifier."""
        _validate_required("filePath", filePath)

        from regions import Region

        if level == "file":
            region = Region.for_file(filePath)
        elif level == "class":
            if not className:
                raise ValueError("className required for class-level region")
            region = Region.for_class(filePath, className)
        elif level == "function":
            if not funcName:
                raise ValueError("funcName required for function-level region")
            region = Region.for_function(filePath, funcName, class_name=className)
        elif level == "lines":
            if lineStart is None or lineEnd is None:
                raise ValueError("lineStart and lineEnd required for lines-level region")
            region = Region.for_lines(
                filePath, lineStart, lineEnd,
                func_name=funcName, class_name=className
            )
        else:
            raise ValueError(f"Invalid level: {level}. Must be one of: file, class, function, lines")

        return {
            "regionId": region.to_string(),
            "level": region.level.value,
        }

    # =========================================================================
    # Belief Revision Tools (powered by py-brs)
    # =========================================================================

    def _get_theory_manager(project_id: str) -> "TheoryManager":
        """Get or create a TheoryManager for a project."""
        from theory import TheoryManager
        return TheoryManager(Path(DATA_DIR) / project_id)

    @server.tool(
        description=(
            "Add an assertion to the synthesis theory with evidence grounding. "
            "Assertions are typed beliefs about code (type, behavior, invariant, contract). "
            "Each assertion must be grounded by evidence (test results, mutation results, etc.)."
        )
    )
    def add_assertion_tool(
        projectId: str,
        assertionType: str,
        content: str,
        evidenceId: str,
        confidence: float = 0.5,
        regionId: Optional[str] = None,
    ) -> dict:
        """Add a typed assertion to the theory."""
        _validate_required("projectId", projectId)
        _validate_required("assertionType", assertionType)
        _validate_required("content", content)
        _validate_required("evidenceId", evidenceId)

        manager = _get_theory_manager(projectId)
        node = manager.add_assertion(
            assertion_type=assertionType,
            content=content,
            evidence_id=evidenceId,
            confidence=confidence,
            region_id=regionId,
        )
        return {
            "node_id": node["id"],
            "assertion_type": assertionType,
            "content": content,
            "confidence": confidence,
            "status": "added",
        }

    @server.tool(
        description=(
            "Contract (remove) an assertion from the theory using AGM contraction. "
            "Strategies: 'entrenchment' (removes target and less-entrenched dependents), "
            "'minimal' (target and edges only), 'full_cascade' (target and all descendants)."
        )
    )
    def contract_assertion_tool(
        projectId: str,
        nodeId: str,
        strategy: str = "entrenchment",
    ) -> dict:
        """Remove an assertion via AGM contraction."""
        _validate_required("projectId", projectId)
        _validate_required("nodeId", nodeId)

        valid_strategies = {"entrenchment", "minimal", "full_cascade"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")

        manager = _get_theory_manager(projectId)
        result = manager.contract_assertion(nodeId, strategy=strategy)

        return {
            "target_node": result.target_node,
            "nodes_removed": list(result.nodes_removed),
            "edges_removed": list(result.edges_removed),
            "strategy": strategy,
            "reason": result.reason,
        }

    @server.tool(
        description=(
            "Get the entrenchment score for an assertion. "
            "Entrenchment measures belief resilience (0.0-1.0, higher = more entrenched). "
            "Highly entrenched beliefs are harder to remove via contraction."
        )
    )
    def get_entrenchment_tool(
        projectId: str,
        nodeId: str,
    ) -> dict:
        """Get entrenchment score for a node."""
        _validate_required("projectId", projectId)
        _validate_required("nodeId", nodeId)

        manager = _get_theory_manager(projectId)
        score = manager.get_entrenchment(nodeId)

        return {
            "node_id": nodeId,
            "entrenchment": score,
            "interpretation": (
                "highly entrenched" if score > 0.7
                else "moderately entrenched" if score > 0.4
                else "weakly entrenched"
            ),
        }

    @server.tool(
        description=(
            "List all assertions in the synthesis theory. "
            "Optionally filter by assertion type (type, behavior, invariant, contract) "
            "or by region ID."
        )
    )
    def list_assertions_tool(
        projectId: str,
        assertionType: Optional[str] = None,
        regionId: Optional[str] = None,
    ) -> dict:
        """List assertions with optional filtering."""
        _validate_required("projectId", projectId)

        manager = _get_theory_manager(projectId)
        assertions = manager.list_assertions(
            assertion_type=assertionType,
            region_id=regionId,
        )

        return {
            "count": len(assertions),
            "assertions": [
                {
                    "id": a["id"],
                    "type": a.get("properties", {}).get("assertion_type"),
                    "content": a.get("properties", {}).get("content"),
                    "confidence": a.get("properties", {}).get("confidence"),
                    "region_id": a.get("properties", {}).get("region_id"),
                }
                for a in assertions
            ],
        }

    @server.tool(
        description=(
            "Get a snapshot of the current synthesis theory. "
            "Returns the world bundle with all nodes, edges, and evidence."
        )
    )
    def get_theory_snapshot_tool(
        projectId: str,
    ) -> dict:
        """Get current theory state."""
        _validate_required("projectId", projectId)

        manager = _get_theory_manager(projectId)
        snapshot = manager.get_theory_snapshot()

        return {
            "domain": snapshot.get("domain_id"),
            "version": snapshot.get("version_label"),
            "node_count": len(snapshot.get("node_ids", [])),
            "edge_count": len(snapshot.get("edge_ids", [])),
            "evidence_count": len(snapshot.get("evidence_ids", [])),
            "created_utc": snapshot.get("created_utc"),
            "notes": snapshot.get("notes"),
        }

    @server.tool(
        description=(
            "Revise the theory by incorporating a new assertion. "
            "If the new assertion contradicts existing beliefs, those are first "
            "contracted (via AGM revision / Levi identity) before adding the new belief."
        )
    )
    def revise_theory_tool(
        projectId: str,
        assertionType: str,
        content: str,
        evidenceId: str,
        confidence: float = 0.5,
        contractionStrategy: str = "entrenchment",
    ) -> dict:
        """Revise theory with new assertion, contracting contradictions."""
        _validate_required("projectId", projectId)
        _validate_required("assertionType", assertionType)
        _validate_required("content", content)
        _validate_required("evidenceId", evidenceId)

        manager = _get_theory_manager(projectId)
        new_hash, contraction = manager.revise_with_assertion(
            assertion_type=assertionType,
            content=content,
            evidence_id=evidenceId,
            confidence=confidence,
            contraction_strategy=contractionStrategy,
        )

        result: Dict[str, Any] = {
            "new_world_hash": new_hash,
            "contraction_performed": contraction is not None,
        }

        if contraction:
            result["contraction_details"] = {
                "nodes_removed": list(contraction.nodes_removed),
                "edges_removed": list(contraction.edges_removed),
                "reason": contraction.reason,
            }

        return result

    # =========================================================================
    # M3: Provenance, Rollback & Failure Analysis Tools
    # =========================================================================

    @server.tool(
        description=(
            "Store a piece of evidence (test result, mutation result, etc.) in the "
            "synthesis theory. Evidence is required to ground assertions."
        )
    )
    def store_evidence_tool(
        projectId: str,
        evidenceId: str,
        citation: str,
        kind: str = "mutation_result",
        reliability: float = 0.7,
    ) -> dict:
        """Store evidence in the theory."""
        _validate_required("projectId", projectId)
        _validate_required("evidenceId", evidenceId)
        _validate_required("citation", citation)

        from brs import Evidence
        import datetime

        evidence = Evidence(
            id=evidenceId,
            citation=citation,
            kind=kind,
            reliability=reliability,
            date=datetime.datetime.utcnow().isoformat() + "Z",
            metadata={},
        )

        manager = _get_theory_manager(projectId)
        stored_id = manager.store_evidence(evidence)

        return {
            "evidence_id": stored_id,
            "kind": kind,
            "reliability": reliability,
            "status": "stored",
        }

    @server.tool(
        description=(
            "Get the provenance DAG summary for a project's synthesis theory. "
            "Shows the history of belief revision operations: expansions, contractions, "
            "revisions, and evidence storage events."
        )
    )
    def get_provenance_tool(projectId: str) -> dict:
        """Get provenance DAG summary."""
        _validate_required("projectId", projectId)

        manager = _get_theory_manager(projectId)
        return manager.get_provenance_summary()

    @server.tool(
        description=(
            "Trace the evidence chain for an assertion. "
            "Returns the list of evidence IDs that ground (support) a given assertion, "
            "answering 'why do we believe this?'."
        )
    )
    def why_believe_tool(projectId: str, assertionId: str) -> dict:
        """Trace evidence chain for an assertion."""
        _validate_required("projectId", projectId)
        _validate_required("assertionId", assertionId)

        manager = _get_theory_manager(projectId)
        evidence_ids = manager.why_believe(assertionId)
        event = manager.when_added(assertionId)

        result: Dict[str, Any] = {
            "assertion_id": assertionId,
            "grounding_evidence_ids": evidence_ids,
            "evidence_count": len(evidence_ids),
        }

        if event:
            result["first_added"] = {
                "timestamp": event.timestamp,
                "event_type": event.event_type.value,
                "reason": event.reason,
            }

        return result

    @server.tool(
        description=(
            "Measure the stability of an assertion. "
            "Returns a score from 0.0 (constantly revised) to 1.0 (never touched). "
            "Unstable assertions may need stronger evidence or reformulation."
        )
    )
    def belief_stability_tool(projectId: str, assertionId: str) -> dict:
        """Measure assertion stability."""
        _validate_required("projectId", projectId)
        _validate_required("assertionId", assertionId)

        manager = _get_theory_manager(projectId)
        score = manager.belief_stability(assertionId)

        return {
            "assertion_id": assertionId,
            "stability": score,
            "interpretation": (
                "very stable" if score > 0.8
                else "stable" if score > 0.5
                else "unstable" if score > 0.2
                else "highly unstable"
            ),
        }

    @server.tool(
        description=(
            "Revert the synthesis theory to a prior world state. "
            "Uses content-addressable storage — no data is lost, only the "
            "current world pointer changes."
        )
    )
    def rollback_to_tool(projectId: str, worldHash: str) -> dict:
        """Rollback to a prior world state."""
        _validate_required("projectId", projectId)
        _validate_required("worldHash", worldHash)

        manager = _get_theory_manager(projectId)
        rollback = manager.get_rollback_manager()
        rollback.rollback_to(worldHash)

        return {
            "status": "rolled_back",
            "target_world_hash": worldHash,
        }

    @server.tool(
        description=(
            "Undo the last N belief revision operations. "
            "Walks backward through the provenance DAG to find the prior "
            "world state, then rolls back to it."
        )
    )
    def undo_last_operations_tool(projectId: str, count: int = 1) -> dict:
        """Undo last N operations."""
        _validate_required("projectId", projectId)

        manager = _get_theory_manager(projectId)
        rollback = manager.get_rollback_manager()
        undone = rollback.undo_last(count)

        return {
            "status": "undone",
            "operations_undone": count,
            "undone_events": [
                {
                    "event_type": e.event_type.value,
                    "assertion_id": e.assertion_id,
                    "timestamp": e.timestamp,
                }
                for e in undone
            ],
        }

    @server.tool(
        description=(
            "Analyze why a synthesis attempt failed. "
            "Classifies the failure mode (type mismatch, overfitting, underfitting, etc.) "
            "and suggests which assertions to contract to fix the issue."
        )
    )
    def analyze_failure_tool(
        projectId: str,
        errorMessage: str = "",
        testPassRate: Optional[float] = None,
        mutationScore: Optional[float] = None,
        regionId: Optional[str] = None,
    ) -> dict:
        """Analyze a synthesis failure."""
        _validate_required("projectId", projectId)

        manager = _get_theory_manager(projectId)
        analysis = manager.analyze_failure(
            error_message=errorMessage,
            test_pass_rate=testPassRate,
            mutation_score=mutationScore,
            region_id=regionId,
        )

        return analysis.to_dict()

    @server.tool(
        description=(
            "List all historical world states for a project's synthesis theory. "
            "Returns world hashes with timestamps and reasons, useful for "
            "understanding theory evolution and choosing rollback targets."
        )
    )
    def list_world_history_tool(projectId: str) -> dict:
        """List historical world states."""
        _validate_required("projectId", projectId)

        manager = _get_theory_manager(projectId)
        rollback = manager.get_rollback_manager()
        history = rollback.list_world_history()

        return {
            "count": len(history),
            "worlds": [
                {
                    "world_hash": h,
                    "timestamp": ts,
                    "reason": reason,
                }
                for h, ts, reason in history
            ],
        }

    # =========================================================================
    # Graph-Spectral Analysis Tools (Phase 2)
    # =========================================================================

    def _require_graph_extras() -> None:
        """Raise a clear error if graph optional dependencies are missing."""
        try:
            import scipy  # noqa: F401
        except ImportError:
            raise ValueError(
                "scipy is required for graph-spectral analysis. "
                "Install with: pip install 'curate-ipsum[graph]'"
            )

    def _require_networkx_extra() -> None:
        """Raise a clear error if networkx is not installed."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            raise ValueError(
                "networkx is required for planarity/reachability analysis. "
                "Install with: pip install 'curate-ipsum[graph]'"
            )

    def _extract_graph(working_directory: str, backend: str = "auto") -> "CallGraph":
        """Extract a call graph from a project directory."""
        from graph import get_extractor, CallGraph as CG

        directory = Path(working_directory)
        if not directory.is_dir():
            raise ValueError(f"Not a valid directory: {working_directory}")

        extractor = get_extractor(backend=backend)
        return extractor.extract_directory(directory)

    @server.tool(
        description=(
            "Extract and analyze the call graph of a Python project. "
            "Returns summary statistics: node count, edge count, SCC count, "
            "connected components, and top-level function list."
        )
    )
    def extract_call_graph(
        workingDirectory: str,
        backend: str = "auto",
    ) -> dict:
        """Extract call graph and return summary statistics."""
        _validate_required("workingDirectory", workingDirectory)

        graph = _extract_graph(workingDirectory, backend)

        sccs = graph.strongly_connected_components()
        non_trivial_sccs = [scc for scc in sccs if len(scc) >= 2]

        # Compute connected components (undirected) via union-find
        parent: Dict[str, str] = {n: n for n in graph.nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in graph.edges:
            if edge.source_id in parent and edge.target_id in parent:
                union(edge.source_id, edge.target_id)

        components: Dict[str, List[str]] = {}
        for node_id in graph.nodes:
            root = find(node_id)
            components.setdefault(root, []).append(node_id)

        return {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "scc_count": len(sccs),
            "non_trivial_scc_count": len(non_trivial_sccs),
            "connected_component_count": len(components),
            "component_sizes": sorted(
                [len(c) for c in components.values()], reverse=True
            ),
            "functions": [
                {"id": n.id, "name": n.name, "kind": n.kind.value}
                for n in sorted(graph.nodes.values(), key=lambda n: n.id)
            ],
        }

    @server.tool(
        description=(
            "Compute Fiedler spectral partitioning of a project's call graph. "
            "Recursively bipartitions the graph using the Fiedler vector (2nd "
            "eigenvector of the graph Laplacian). Returns a partition tree with "
            "node assignments and algebraic connectivity (λ₂) values."
        )
    )
    def compute_partitioning(
        workingDirectory: str,
        min_partition_size: int = 3,
        max_depth: int = 10,
    ) -> dict:
        """Compute spectral partitioning and return the partition tree."""
        _validate_required("workingDirectory", workingDirectory)
        _require_graph_extras()

        from graph import GraphPartitioner

        graph = _extract_graph(workingDirectory)
        partitioner = GraphPartitioner(
            min_partition_size=min_partition_size,
            max_depth=max_depth,
        )
        root = partitioner.partition(graph)
        leaves = GraphPartitioner.get_leaf_partitions(root)

        def _serialize_partition(p) -> dict:
            result = {
                "id": p.id,
                "size": p.size,
                "depth": p.depth,
                "is_leaf": p.is_leaf,
                "node_ids": sorted(p.node_ids),
            }
            if p.fiedler_value is not None:
                result["fiedler_value"] = round(p.fiedler_value, 6)
            if p.children is not None:
                result["children"] = [
                    _serialize_partition(p.children[0]),
                    _serialize_partition(p.children[1]),
                ]
            return result

        return {
            "total_nodes": root.size,
            "leaf_partition_count": len(leaves),
            "leaf_sizes": sorted([l.size for l in leaves], reverse=True),
            "partition_tree": _serialize_partition(root),
        }

    @server.tool(
        description=(
            "Query reachability between two functions in a project's call graph. "
            "Uses Kameda O(1) index for planar subgraphs with BFS fallback for "
            "non-planar edges. Returns whether the source can reach the target, "
            "the method used, and the path if reachable via BFS."
        )
    )
    def query_reachability(
        workingDirectory: str,
        source_function: str,
        target_function: str,
    ) -> dict:
        """Query reachability between two functions."""
        _validate_required("workingDirectory", workingDirectory)
        _validate_required("source_function", source_function)
        _validate_required("target_function", target_function)
        _require_graph_extras()
        _require_networkx_extra()

        from graph import check_planarity, KamedaIndex
        from graph.partitioner import augment_partition, GraphPartitioner

        graph = _extract_graph(workingDirectory)

        # Validate both functions exist
        if source_function not in graph.nodes:
            # Try fuzzy match: search by short name
            matches = [
                nid for nid, n in graph.nodes.items()
                if n.name == source_function or nid.endswith(f".{source_function}")
            ]
            if len(matches) == 1:
                source_function = matches[0]
            elif len(matches) > 1:
                return {
                    "error": f"Ambiguous source function '{source_function}'. Matches: {matches}",
                    "reachable": None,
                }
            else:
                return {
                    "error": f"Source function '{source_function}' not found in call graph.",
                    "reachable": None,
                    "available_functions": sorted(graph.nodes.keys())[:50],
                }

        if target_function not in graph.nodes:
            matches = [
                nid for nid, n in graph.nodes.items()
                if n.name == target_function or nid.endswith(f".{target_function}")
            ]
            if len(matches) == 1:
                target_function = matches[0]
            elif len(matches) > 1:
                return {
                    "error": f"Ambiguous target function '{target_function}'. Matches: {matches}",
                    "reachable": None,
                }
            else:
                return {
                    "error": f"Target function '{target_function}' not found in call graph.",
                    "reachable": None,
                    "available_functions": sorted(graph.nodes.keys())[:50],
                }

        # BFS-based path finding (always available, used as ground truth)
        bfs_reachable = target_function in graph.reachable_from(source_function)
        bfs_path: Optional[List[str]] = None
        if bfs_reachable:
            # Reconstruct path via BFS
            from collections import deque
            visited: Dict[str, Optional[str]] = {source_function: None}
            queue = deque([source_function])
            while queue:
                current = queue.popleft()
                if current == target_function:
                    break
                for callee in graph.get_callees(current):
                    if callee not in visited:
                        visited[callee] = current
                        queue.append(callee)
            if target_function in visited:
                path: List[str] = []
                node = target_function
                while node is not None:
                    path.append(node)
                    node = visited[node]
                bfs_path = list(reversed(path))

        # Try Kameda index for O(1) check
        method = "bfs"
        try:
            # Condense SCCs first (Kameda needs a DAG)
            condensed = graph.condensation()
            planarity_result = check_planarity(condensed)

            kameda_index = KamedaIndex.build(
                planarity_result.planar_subgraph,
                embedding=planarity_result.embedding,
                non_planar_edges=planarity_result.non_planar_edges,
            )

            # Map original function IDs to their SCC IDs
            sccs = graph.strongly_connected_components()
            node_to_scc: Dict[str, str] = {}
            for i, scc in enumerate(sccs):
                for n in scc:
                    node_to_scc[n] = f"scc_{i}"

            src_scc = node_to_scc.get(source_function)
            tgt_scc = node_to_scc.get(target_function)

            if src_scc and tgt_scc:
                if src_scc == tgt_scc:
                    kameda_reachable = True
                else:
                    kameda_reachable = kameda_index.reaches(src_scc, tgt_scc)
                method = "kameda"
        except (ValueError, ImportError):
            # Kameda build failed — fall back to BFS (already computed)
            pass

        return {
            "source": source_function,
            "target": target_function,
            "reachable": bfs_reachable,
            "method": method,
            "path": bfs_path,
        }

    @server.tool(
        description=(
            "Get the hierarchical decomposition of a project's call graph. "
            "Alternates between SCC condensation and Fiedler spectral partitioning "
            "to produce a tree representing the project's modular structure."
        )
    )
    def get_hierarchy(workingDirectory: str) -> dict:
        """Get hierarchical decomposition of the call graph."""
        _validate_required("workingDirectory", workingDirectory)
        _require_graph_extras()

        from graph import HierarchyBuilder

        graph = _extract_graph(workingDirectory)
        builder = HierarchyBuilder()
        root = builder.build(graph)

        summary = builder.summary(root)
        leaf_groups = builder.flatten(root)

        summary["leaf_group_count"] = len(leaf_groups)
        summary["leaf_groups"] = [
            {"size": len(group), "members": sorted(group)}
            for group in sorted(leaf_groups, key=len, reverse=True)
        ]

        return summary

    @server.tool(
        description=(
            "Find which partition a function belongs to in the Fiedler partition tree. "
            "Returns the partition ID, sibling functions in the same partition, "
            "and the entry/exit points of that partition."
        )
    )
    def find_function_partition(
        workingDirectory: str,
        function_name: str,
    ) -> dict:
        """Find which partition a function belongs to."""
        _validate_required("workingDirectory", workingDirectory)
        _validate_required("function_name", function_name)
        _require_graph_extras()

        from graph import GraphPartitioner

        graph = _extract_graph(workingDirectory)

        # Resolve function name
        resolved = function_name
        if function_name not in graph.nodes:
            matches = [
                nid for nid, n in graph.nodes.items()
                if n.name == function_name or nid.endswith(f".{function_name}")
            ]
            if len(matches) == 1:
                resolved = matches[0]
            elif len(matches) > 1:
                return {
                    "error": f"Ambiguous function name '{function_name}'. Matches: {matches}",
                }
            else:
                return {
                    "error": f"Function '{function_name}' not found in call graph.",
                    "available_functions": sorted(graph.nodes.keys())[:50],
                }

        partitioner = GraphPartitioner()
        root = partitioner.partition(graph)
        leaf = GraphPartitioner.find_partition(root, resolved)

        if leaf is None:
            return {
                "error": f"Function '{resolved}' not found in any partition.",
            }

        # Find entry/exit points within the partition
        partition_nodes = leaf.node_ids
        has_internal_incoming: set = set()
        has_internal_outgoing: set = set()
        for edge in graph.edges:
            if edge.source_id in partition_nodes and edge.target_id in partition_nodes:
                has_internal_incoming.add(edge.target_id)
                has_internal_outgoing.add(edge.source_id)

        entry_points = sorted(partition_nodes - has_internal_incoming)
        exit_points = sorted(partition_nodes - has_internal_outgoing)

        return {
            "function": resolved,
            "partition_id": leaf.id,
            "partition_size": leaf.size,
            "partition_depth": leaf.depth,
            "siblings": sorted(nid for nid in leaf.node_ids if nid != resolved),
            "entry_points": entry_points,
            "exit_points": exit_points,
            "fiedler_value": round(leaf.fiedler_value, 6) if leaf.fiedler_value else None,
        }

    return server


def main() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
