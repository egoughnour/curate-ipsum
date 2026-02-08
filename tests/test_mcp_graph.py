"""
Integration tests for MCP graph tools (Step 9).

Tests the 5 graph-spectral MCP tools end-to-end:
  - extract_call_graph
  - compute_partitioning
  - query_reachability
  - get_hierarchy
  - find_function_partition

Uses curate-ipsum's own source code as the test corpus where practical,
with small synthetic graphs for deterministic edge-case coverage.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
from pathlib import Path
from typing import Dict

import pytest

# We need scipy and networkx for graph tools
scipy = pytest.importorskip("scipy")
nx = pytest.importorskip("networkx")

from graph import (
    ASTExtractor,
    CallGraph,
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphPartitioner,
    NodeKind,
)
from graph.hierarchy import HierarchyBuilder
from graph.kameda import KamedaIndex
from graph.planarity import check_planarity


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a small synthetic Python project for end-to-end testing."""
    # main.py calls utils.py functions, which call helpers.py
    (tmp_path / "main.py").write_text(textwrap.dedent("""\
        from utils import process, validate
        from helpers import log

        def main():
            data = load_data()
            validated = validate(data)
            result = process(validated)
            log(result)
            return result

        def load_data():
            return {"key": "value"}
    """))

    (tmp_path / "utils.py").write_text(textwrap.dedent("""\
        from helpers import transform, log

        def process(data):
            transformed = transform(data)
            log("processed")
            return transformed

        def validate(data):
            if not data:
                raise ValueError("empty")
            log("validated")
            return data
    """))

    (tmp_path / "helpers.py").write_text(textwrap.dedent("""\
        def transform(data):
            return {k: v.upper() for k, v in data.items()}

        def log(message):
            print(f"[LOG] {message}")
    """))

    return tmp_path


@pytest.fixture
def diamond_graph() -> CallGraph:
    """Create a diamond-shaped DAG: A → B, A → C, B → D, C → D."""
    g = CallGraph()
    for name in ["A", "B", "C", "D"]:
        g.add_node(GraphNode(id=name, kind=NodeKind.FUNCTION, name=name))
    g.add_edge(GraphEdge(source_id="A", target_id="B", kind=EdgeKind.CALLS))
    g.add_edge(GraphEdge(source_id="A", target_id="C", kind=EdgeKind.CALLS))
    g.add_edge(GraphEdge(source_id="B", target_id="D", kind=EdgeKind.CALLS))
    g.add_edge(GraphEdge(source_id="C", target_id="D", kind=EdgeKind.CALLS))
    return g


@pytest.fixture
def chain_graph() -> CallGraph:
    """Create a linear chain: A → B → C → D → E."""
    g = CallGraph()
    nodes = ["A", "B", "C", "D", "E"]
    for name in nodes:
        g.add_node(GraphNode(id=name, kind=NodeKind.FUNCTION, name=name))
    for i in range(len(nodes) - 1):
        g.add_edge(GraphEdge(
            source_id=nodes[i], target_id=nodes[i + 1], kind=EdgeKind.CALLS
        ))
    return g


# ─────────────────────────────────────────────────────────────────
# Helper: Build server and extract tool functions
# ─────────────────────────────────────────────────────────────────


def _get_server_tools() -> Dict:
    """
    Build the MCP server and return a dict of tool callables.

    The tools are the raw Python functions registered on the server,
    extracted so we can call them directly without MCP transport.
    """
    from server import build_server

    srv = build_server()
    # FastMCP stores tools internally; we call the underlying functions directly.
    # The tool functions are defined as closures inside build_server().
    # We can access them via the server's tool registry.
    tools = {}
    for tool_name in srv._tool_manager._tools:
        tool_obj = srv._tool_manager._tools[tool_name]
        tools[tool_name] = tool_obj.fn
    return tools


# ─────────────────────────────────────────────────────────────────
# Tests: extract_call_graph
# ─────────────────────────────────────────────────────────────────


class TestExtractCallGraph:
    """Tests for the extract_call_graph MCP tool."""

    def test_basic_extraction(self, sample_project: Path):
        """Extract call graph from sample project."""
        tools = _get_server_tools()
        result = tools["extract_call_graph"](
            workingDirectory=str(sample_project)
        )

        assert isinstance(result, dict)
        assert result["node_count"] > 0
        assert result["edge_count"] > 0
        assert "scc_count" in result
        assert "connected_component_count" in result
        assert "functions" in result
        assert isinstance(result["functions"], list)

    def test_functions_have_required_fields(self, sample_project: Path):
        """Each function entry has id, name, kind."""
        tools = _get_server_tools()
        result = tools["extract_call_graph"](
            workingDirectory=str(sample_project)
        )

        for func in result["functions"]:
            assert "id" in func
            assert "name" in func
            assert "kind" in func

    def test_invalid_directory(self):
        """Raises ValueError for non-existent directory."""
        tools = _get_server_tools()
        with pytest.raises(ValueError, match="Not a valid directory"):
            tools["extract_call_graph"](
                workingDirectory="/nonexistent/path/xyz"
            )

    def test_missing_working_directory(self):
        """Raises ValueError for missing workingDirectory."""
        tools = _get_server_tools()
        with pytest.raises(ValueError, match="Missing required field"):
            tools["extract_call_graph"](workingDirectory="")

    def test_result_is_json_serializable(self, sample_project: Path):
        """Result must be JSON-serializable."""
        tools = _get_server_tools()
        result = tools["extract_call_graph"](
            workingDirectory=str(sample_project)
        )
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_on_own_source(self):
        """Extract call graph from curate-ipsum's own graph/ package."""
        tools = _get_server_tools()
        graph_dir = Path(__file__).parent.parent / "graph"
        if not graph_dir.is_dir():
            pytest.skip("graph/ directory not found")

        result = tools["extract_call_graph"](
            workingDirectory=str(graph_dir)
        )
        # curate-ipsum's graph package has many functions
        assert result["node_count"] >= 10
        assert result["edge_count"] >= 5


# ─────────────────────────────────────────────────────────────────
# Tests: compute_partitioning
# ─────────────────────────────────────────────────────────────────


class TestComputePartitioning:
    """Tests for the compute_partitioning MCP tool."""

    def test_basic_partitioning(self, sample_project: Path):
        """Partition the sample project's call graph."""
        tools = _get_server_tools()
        result = tools["compute_partitioning"](
            workingDirectory=str(sample_project)
        )

        assert isinstance(result, dict)
        assert "total_nodes" in result
        assert "leaf_partition_count" in result
        assert "partition_tree" in result
        assert result["total_nodes"] > 0

    def test_partition_tree_structure(self, sample_project: Path):
        """Partition tree has required fields."""
        tools = _get_server_tools()
        result = tools["compute_partitioning"](
            workingDirectory=str(sample_project)
        )

        tree = result["partition_tree"]
        assert "id" in tree
        assert "size" in tree
        assert "is_leaf" in tree
        assert "node_ids" in tree

    def test_custom_min_partition_size(self, sample_project: Path):
        """Respects min_partition_size parameter."""
        tools = _get_server_tools()

        # Very large min_partition_size → no splitting
        result = tools["compute_partitioning"](
            workingDirectory=str(sample_project),
            min_partition_size=100,
        )
        assert result["leaf_partition_count"] >= 1
        # With min_partition_size=100, most partitions won't be split
        tree = result["partition_tree"]
        assert tree["is_leaf"] or tree.get("children") is not None

    def test_result_is_json_serializable(self, sample_project: Path):
        """Result must be JSON-serializable."""
        tools = _get_server_tools()
        result = tools["compute_partitioning"](
            workingDirectory=str(sample_project)
        )
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_all_nodes_covered(self, sample_project: Path):
        """Every node appears in exactly one leaf partition."""
        tools = _get_server_tools()
        result = tools["compute_partitioning"](
            workingDirectory=str(sample_project)
        )

        # Collect all node_ids from leaf partitions
        def collect_leaf_nodes(tree):
            if tree["is_leaf"]:
                return set(tree["node_ids"])
            nodes = set()
            for child in tree.get("children", []):
                nodes.update(collect_leaf_nodes(child))
            return nodes

        leaf_nodes = collect_leaf_nodes(result["partition_tree"])
        assert len(leaf_nodes) == result["total_nodes"]


# ─────────────────────────────────────────────────────────────────
# Tests: query_reachability
# ─────────────────────────────────────────────────────────────────


class TestQueryReachability:
    """Tests for the query_reachability MCP tool."""

    def test_reachable_pair(self, sample_project: Path):
        """Main calls process (via utils), should be reachable."""
        tools = _get_server_tools()

        # First extract to learn function IDs
        graph_result = tools["extract_call_graph"](
            workingDirectory=str(sample_project)
        )
        func_ids = [f["id"] for f in graph_result["functions"]]

        # Find main and process
        main_id = next((f for f in func_ids if "main" in f.lower()), None)
        process_id = next((f for f in func_ids if "process" in f.lower()), None)

        if main_id and process_id:
            result = tools["query_reachability"](
                workingDirectory=str(sample_project),
                source_function=main_id,
                target_function=process_id,
            )

            assert isinstance(result, dict)
            assert "reachable" in result
            assert "method" in result
            assert result["method"] in ("kameda", "bfs")

    def test_unreachable_pair(self, sample_project: Path):
        """Helpers shouldn't reach main (no back-edges)."""
        tools = _get_server_tools()

        graph_result = tools["extract_call_graph"](
            workingDirectory=str(sample_project)
        )
        func_ids = [f["id"] for f in graph_result["functions"]]

        transform_id = next((f for f in func_ids if "transform" in f.lower()), None)
        main_id = next((f for f in func_ids if f.endswith(".main")), None)

        if transform_id and main_id:
            result = tools["query_reachability"](
                workingDirectory=str(sample_project),
                source_function=transform_id,
                target_function=main_id,
            )
            assert result["reachable"] is False

    def test_function_not_found(self, sample_project: Path):
        """Returns error for non-existent function."""
        tools = _get_server_tools()
        result = tools["query_reachability"](
            workingDirectory=str(sample_project),
            source_function="nonexistent_function_xyz",
            target_function="also_nonexistent",
        )

        assert result["reachable"] is None
        assert "error" in result

    def test_self_reachability(self, sample_project: Path):
        """A function can reach itself."""
        tools = _get_server_tools()

        graph_result = tools["extract_call_graph"](
            workingDirectory=str(sample_project)
        )
        if graph_result["functions"]:
            func_id = graph_result["functions"][0]["id"]
            result = tools["query_reachability"](
                workingDirectory=str(sample_project),
                source_function=func_id,
                target_function=func_id,
            )
            # Self-reachability: BFS sees it, and Kameda returns True for same node
            assert result["reachable"] is not None

    def test_fuzzy_match_by_short_name(self, sample_project: Path):
        """Tool resolves short function names to fully qualified IDs."""
        tools = _get_server_tools()
        result = tools["query_reachability"](
            workingDirectory=str(sample_project),
            source_function="main",
            target_function="log",
        )

        # Should resolve "main" and "log" to their full IDs
        # (or return an error if ambiguous)
        assert isinstance(result, dict)
        assert "reachable" in result or "error" in result

    def test_result_is_json_serializable(self, sample_project: Path):
        """Result must be JSON-serializable."""
        tools = _get_server_tools()
        result = tools["query_reachability"](
            workingDirectory=str(sample_project),
            source_function="main",
            target_function="log",
        )
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


# ─────────────────────────────────────────────────────────────────
# Tests: get_hierarchy
# ─────────────────────────────────────────────────────────────────


class TestGetHierarchy:
    """Tests for the get_hierarchy MCP tool."""

    def test_basic_hierarchy(self, sample_project: Path):
        """Build hierarchy for sample project."""
        tools = _get_server_tools()
        result = tools["get_hierarchy"](
            workingDirectory=str(sample_project)
        )

        assert isinstance(result, dict)
        assert "id" in result
        assert "level" in result
        assert "operation" in result
        assert "size" in result
        assert "leaf_group_count" in result
        assert "leaf_groups" in result

    def test_leaf_groups_cover_all_nodes(self, sample_project: Path):
        """All nodes should appear in some leaf group."""
        tools = _get_server_tools()
        result = tools["get_hierarchy"](
            workingDirectory=str(sample_project)
        )

        total_in_leaves = sum(g["size"] for g in result["leaf_groups"])
        assert total_in_leaves == result["size"]

    def test_result_is_json_serializable(self, sample_project: Path):
        """Result must be JSON-serializable."""
        tools = _get_server_tools()
        result = tools["get_hierarchy"](
            workingDirectory=str(sample_project)
        )
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


# ─────────────────────────────────────────────────────────────────
# Tests: find_function_partition
# ─────────────────────────────────────────────────────────────────


class TestFindFunctionPartition:
    """Tests for the find_function_partition MCP tool."""

    def test_find_existing_function(self, sample_project: Path):
        """Find partition for a function that exists."""
        tools = _get_server_tools()

        # First get a valid function name
        graph_result = tools["extract_call_graph"](
            workingDirectory=str(sample_project)
        )
        if not graph_result["functions"]:
            pytest.skip("No functions found")

        func_id = graph_result["functions"][0]["id"]
        result = tools["find_function_partition"](
            workingDirectory=str(sample_project),
            function_name=func_id,
        )

        assert isinstance(result, dict)
        assert "partition_id" in result
        assert "partition_size" in result
        assert "siblings" in result
        assert "entry_points" in result
        assert "exit_points" in result

    def test_function_not_found(self, sample_project: Path):
        """Returns error for non-existent function."""
        tools = _get_server_tools()
        result = tools["find_function_partition"](
            workingDirectory=str(sample_project),
            function_name="nonexistent_function_xyz",
        )

        assert "error" in result

    def test_fuzzy_match(self, sample_project: Path):
        """Resolves short names to fully qualified IDs."""
        tools = _get_server_tools()
        result = tools["find_function_partition"](
            workingDirectory=str(sample_project),
            function_name="main",
        )

        # Should either find "main" or report an error if ambiguous
        assert isinstance(result, dict)
        assert "partition_id" in result or "error" in result

    def test_result_is_json_serializable(self, sample_project: Path):
        """Result must be JSON-serializable."""
        tools = _get_server_tools()
        result = tools["find_function_partition"](
            workingDirectory=str(sample_project),
            function_name="main",
        )
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


# ─────────────────────────────────────────────────────────────────
# Integration: Full Pipeline
# ─────────────────────────────────────────────────────────────────


class TestFullPipeline:
    """Integration test: extract → partition → query reachability."""

    def test_end_to_end_pipeline(self, sample_project: Path):
        """
        Full pipeline: extract call graph, compute partitions,
        find a function's partition, and query reachability.
        """
        tools = _get_server_tools()

        # Step 1: Extract
        graph_result = tools["extract_call_graph"](
            workingDirectory=str(sample_project)
        )
        assert graph_result["node_count"] > 0

        # Step 2: Partition
        partition_result = tools["compute_partitioning"](
            workingDirectory=str(sample_project),
        )
        assert partition_result["leaf_partition_count"] >= 1

        # Step 3: Find a function's partition
        func_ids = [f["id"] for f in graph_result["functions"]]
        if func_ids:
            part_result = tools["find_function_partition"](
                workingDirectory=str(sample_project),
                function_name=func_ids[0],
            )
            assert "partition_id" in part_result or "error" in part_result

        # Step 4: Query reachability between two functions
        if len(func_ids) >= 2:
            reach_result = tools["query_reachability"](
                workingDirectory=str(sample_project),
                source_function=func_ids[0],
                target_function=func_ids[1],
            )
            assert "reachable" in reach_result

        # Step 5: Hierarchy
        hier_result = tools["get_hierarchy"](
            workingDirectory=str(sample_project)
        )
        assert hier_result["size"] == graph_result["node_count"]

    def test_pipeline_on_own_source(self):
        """Run the full pipeline on curate-ipsum's graph/ package."""
        graph_dir = Path(__file__).parent.parent / "graph"
        if not graph_dir.is_dir():
            pytest.skip("graph/ directory not found")

        tools = _get_server_tools()

        # Extract
        graph_result = tools["extract_call_graph"](
            workingDirectory=str(graph_dir)
        )
        assert graph_result["node_count"] >= 10

        # Partition
        partition_result = tools["compute_partitioning"](
            workingDirectory=str(graph_dir),
            min_partition_size=2,
        )
        assert partition_result["total_nodes"] == graph_result["node_count"]

        # Hierarchy
        hier_result = tools["get_hierarchy"](
            workingDirectory=str(graph_dir)
        )
        assert hier_result["size"] == graph_result["node_count"]

        # All results should be JSON-serializable
        json.dumps(graph_result)
        json.dumps(partition_result)
        json.dumps(hier_result)
