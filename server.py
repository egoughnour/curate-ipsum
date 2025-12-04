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

    @server.tool()
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

    @server.tool()
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

    @server.tool()
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

    @server.tool()
    def get_run_history_tool(projectId: str, regionId: Optional[str] = None, limit: Optional[int] = None) -> dict:
        _validate_required("projectId", projectId)
        history = history_tool(projectId=projectId, regionId=regionId, limit=limit)
        return _json_payload(history)

    @server.tool()
    def get_region_metrics_tool(projectId: str, commitSha: str, regionId: str) -> dict:
        _validate_required("projectId", projectId)
        _validate_required("commitSha", commitSha)
        _validate_required("regionId", regionId)
        metrics = region_metrics_tool(projectId=projectId, commitSha=commitSha, regionId=regionId)
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
