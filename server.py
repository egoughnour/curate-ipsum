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
            description="Run mutation tests (e.g., Stryker) and return summarized mutation statistics."
    )
    async def run_mutation_tests_tool(
        projectId: str,
        commitSha: str,
        command: str,
        workingDirectory: str,
        regionId: Optional[str] = None,
        tool: str = "stryker",
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
