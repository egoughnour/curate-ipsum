from __future__ import annotations

import logging
import os
from typing import Optional

from tools import (
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
        extended_timeout: Optional[float] = None,
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
            extended_timeout=extended_timeout,
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
        extended_timeout: Optional[float] = None,
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
            extended_timeout=extended_timeout,
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
        extended_timeout: Optional[float] = None,
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
            extended_timeout=extended_timeout,
        )
        return _json_payload(run)

    @server.tool(
            description="Return recent unit, integration, and mutation runs for a project and optional region."
    )
    def get_run_history_tool(
        projectId: str, regionId: Optional[str] = None, limit: Optional[int] = None, extended_timeout: Optional[float] = None
    ) -> dict:
        _validate_required("projectId", projectId)
        history = history_tool(projectId=projectId, regionId=regionId, limit=limit, extended_timeout=extended_timeout)
        return _json_payload(history)

    @server.tool(
            description="Compute PID-like metrics and mutation score for a specific region within a project."
                 )
    def get_region_metrics_tool(
        projectId: str, commitSha: str, regionId: str, extended_timeout: Optional[float] = None
    ) -> dict:
        _validate_required("projectId", projectId)
        _validate_required("commitSha", commitSha)
        _validate_required("regionId", regionId)
        metrics = region_metrics_tool(
            projectId=projectId, commitSha=commitSha, regionId=regionId, extended_timeout=extended_timeout
        )
        return _json_payload(metrics)

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
