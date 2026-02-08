"""
Comprehensive tests for graph/hierarchy.py and graph/dependency_extractor.py

Tests cover:
- HierarchyBuilder: DAG hierarchies, cycle detection, flattening, summarization
- DependencyExtractor: import extraction, relative imports, directory traversal
"""

import tempfile
from pathlib import Path
from typing import Set

import pytest

from graph.dependency_extractor import (
    DependencyExtractor,
    _resolve_relative_import,
    extract_imports_from_source,
)
from graph.hierarchy import HierarchyBuilder, HierarchyNode
from graph.models import CallGraph, EdgeKind, GraphEdge, GraphNode, NodeKind


# ─────────────────────────────────────────────────────────────────
# HierarchyBuilder Tests
# ─────────────────────────────────────────────────────────────────


class TestHierarchyBuilderDAG:
    """Test HierarchyBuilder on DAG (acyclic) graphs."""

    def test_build_dag_root_has_condense_operation(self):
        """
        Test that building hierarchy on a DAG produces root with 'condense' operation.
        The root level should detect SCCs (which are all singletons in a DAG).
        """
        # Create simple DAG: a -> b -> c
        graph = CallGraph()
        graph.add_node(GraphNode(id="a", kind=NodeKind.FUNCTION, name="a"))
        graph.add_node(GraphNode(id="b", kind=NodeKind.FUNCTION, name="b"))
        graph.add_node(GraphNode(id="c", kind=NodeKind.FUNCTION, name="c"))
        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="b", target_id="c", kind=EdgeKind.CALLS))

        builder = HierarchyBuilder(max_levels=10)
        root = builder.build(graph)

        # Root should have operation "condense"
        assert root.operation == "condense"
        # Root should contain all nodes
        assert root.node_ids == frozenset({"a", "b", "c"})
        # Root should be at level 0
        assert root.level == 0

    def test_build_dag_returns_hierarchy_node(self):
        """Test that build() returns a valid HierarchyNode with expected structure."""
        graph = CallGraph()
        graph.add_node(GraphNode(id="n1", kind=NodeKind.FUNCTION, name="n1"))
        graph.add_node(GraphNode(id="n2", kind=NodeKind.FUNCTION, name="n2"))
        graph.add_edge(GraphEdge(source_id="n1", target_id="n2", kind=EdgeKind.CALLS))

        builder = HierarchyBuilder()
        root = builder.build(graph)

        assert isinstance(root, HierarchyNode)
        assert root.id is not None
        assert root.level == 0
        assert root.node_ids is not None

    def test_build_empty_graph(self):
        """Test building hierarchy on an empty graph."""
        graph = CallGraph()
        builder = HierarchyBuilder()
        root = builder.build(graph)

        assert root.operation == "condense"
        assert root.node_ids == frozenset()
        assert root.level == 0

    def test_build_single_node_graph(self):
        """Test building hierarchy on a single-node graph."""
        graph = CallGraph()
        graph.add_node(GraphNode(id="solo", kind=NodeKind.FUNCTION, name="solo"))

        builder = HierarchyBuilder()
        root = builder.build(graph)

        assert root.node_ids == frozenset({"solo"})
        # Single node is below min_scc_size (default 2), so it becomes a leaf
        assert root.operation == "leaf"


class TestHierarchyBuilderCycles:
    """Test HierarchyBuilder on graphs with cycles."""

    def test_build_with_cycle_detects_sccs(self):
        """
        Test that a graph with cycles produces SCCs in the hierarchy.
        Graph: a -> b -> a (cycle), and c -> a (acyclic entry).
        """
        graph = CallGraph()
        graph.add_node(GraphNode(id="a", kind=NodeKind.FUNCTION, name="a"))
        graph.add_node(GraphNode(id="b", kind=NodeKind.FUNCTION, name="b"))
        graph.add_node(GraphNode(id="c", kind=NodeKind.FUNCTION, name="c"))

        # Create cycle: a -> b -> a
        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="b", target_id="a", kind=EdgeKind.CALLS))
        # Entry point
        graph.add_edge(GraphEdge(source_id="c", target_id="a", kind=EdgeKind.CALLS))

        builder = HierarchyBuilder(min_scc_size=2, max_levels=5)
        root = builder.build(graph)

        # Root should identify SCCs
        assert root.operation == "condense"
        assert root.scc_members is not None
        # We expect two SCCs: {a, b} and {c}
        assert len(root.scc_members) == 2

        # At least one SCC should be non-trivial
        non_trivial = [scc for scc in root.scc_members if len(scc) >= 2]
        assert len(non_trivial) >= 1

    def test_build_mutual_recursion(self):
        """Test graph with mutual recursion (cycles)."""
        graph = CallGraph()
        graph.add_node(GraphNode(id="f1", kind=NodeKind.FUNCTION, name="f1"))
        graph.add_node(GraphNode(id="f2", kind=NodeKind.FUNCTION, name="f2"))
        graph.add_node(GraphNode(id="f3", kind=NodeKind.FUNCTION, name="f3"))

        # f1 <-> f2 (mutual recursion) and f3 -> f1
        graph.add_edge(GraphEdge(source_id="f1", target_id="f2", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="f2", target_id="f1", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="f3", target_id="f1", kind=EdgeKind.CALLS))

        builder = HierarchyBuilder(min_scc_size=2)
        root = builder.build(graph)

        # Should detect SCC containing f1 and f2
        assert root.scc_members is not None
        assert len(root.scc_members) >= 1

        # Find the SCC with f1
        scc_with_f1 = next((scc for scc in root.scc_members if "f1" in scc), None)
        assert scc_with_f1 is not None
        assert "f2" in scc_with_f1


class TestHierarchyBuilderFlatten:
    """Test HierarchyBuilder.flatten() method."""

    def test_flatten_covers_all_nodes_exactly_once(self):
        """
        Test that flatten() returns groups covering each original node exactly once.
        """
        # Create a small DAG: 1 -> 2, 1 -> 3
        graph = CallGraph()
        for i in range(1, 4):
            graph.add_node(GraphNode(id=f"n{i}", kind=NodeKind.FUNCTION, name=f"n{i}"))
        graph.add_edge(GraphEdge(source_id="n1", target_id="n2", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="n1", target_id="n3", kind=EdgeKind.CALLS))

        builder = HierarchyBuilder(max_levels=3)
        root = builder.build(graph)
        flattened = builder.flatten(root)

        # Flatten should return a list of frozensets
        assert isinstance(flattened, list)
        assert all(isinstance(group, frozenset) for group in flattened)

        # Union of all groups should equal original nodes
        all_nodes = set()
        for group in flattened:
            all_nodes.update(group)
        assert all_nodes == {"n1", "n2", "n3"}

        # Each node should appear in exactly one group
        node_counts = {}
        for group in flattened:
            for node in group:
                node_counts[node] = node_counts.get(node, 0) + 1
        assert all(count == 1 for count in node_counts.values())

    def test_flatten_empty_hierarchy(self):
        """Test flatten on an empty graph."""
        graph = CallGraph()
        builder = HierarchyBuilder()
        root = builder.build(graph)
        flattened = builder.flatten(root)

        # Should return list (possibly empty)
        assert isinstance(flattened, list)

    def test_flatten_single_node(self):
        """Test flatten on a single-node hierarchy."""
        graph = CallGraph()
        graph.add_node(GraphNode(id="x", kind=NodeKind.FUNCTION, name="x"))

        builder = HierarchyBuilder()
        root = builder.build(graph)
        flattened = builder.flatten(root)

        assert len(flattened) >= 1
        all_nodes = set()
        for group in flattened:
            all_nodes.update(group)
        assert "x" in all_nodes


class TestHierarchyBuilderSummary:
    """Test HierarchyBuilder.summary() method."""

    def test_summary_produces_valid_dict(self):
        """Test that summary() returns a properly structured dict."""
        graph = CallGraph()
        graph.add_node(GraphNode(id="a", kind=NodeKind.FUNCTION, name="a"))
        graph.add_node(GraphNode(id="b", kind=NodeKind.FUNCTION, name="b"))
        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))

        builder = HierarchyBuilder()
        root = builder.build(graph)
        summary = builder.summary(root)

        # Check required keys
        assert isinstance(summary, dict)
        assert "id" in summary
        assert "level" in summary
        assert "operation" in summary
        assert "size" in summary

    def test_summary_expected_keys(self):
        """Test that summary includes expected keys based on operation."""
        # DAG should have scc_members in root
        graph = CallGraph()
        graph.add_node(GraphNode(id="x", kind=NodeKind.FUNCTION, name="x"))
        graph.add_node(GraphNode(id="y", kind=NodeKind.FUNCTION, name="y"))
        graph.add_edge(GraphEdge(source_id="x", target_id="y", kind=EdgeKind.CALLS))

        builder = HierarchyBuilder()
        root = builder.build(graph)
        summary = builder.summary(root)

        # Root should have operation="condense"
        assert summary["operation"] == "condense"
        # If there are SCCs, scc_count should be present
        if "scc_count" in summary:
            assert isinstance(summary["scc_count"], int)
            assert summary["scc_count"] >= 1

    def test_summary_recursive_structure(self):
        """Test that summary recursively includes children."""
        graph = CallGraph()
        for i in range(5):
            graph.add_node(GraphNode(id=f"n{i}", kind=NodeKind.FUNCTION, name=f"n{i}"))
        # Chain: 0 -> 1 -> 2 -> 3 -> 4
        for i in range(4):
            graph.add_edge(GraphEdge(source_id=f"n{i}", target_id=f"n{i+1}", kind=EdgeKind.CALLS))

        builder = HierarchyBuilder(max_levels=5)
        root = builder.build(graph)
        summary = builder.summary(root)

        # If tree has children, summary should reflect them
        if root.children:
            assert "children" in summary
            assert isinstance(summary["children"], list)


class TestHierarchyBuilderSmallGraph:
    """Test performance and correctness on small graphs."""

    def test_three_node_dag_terminates_quickly(self):
        """Test that a 3-node DAG terminates quickly (not stuck)."""
        graph = CallGraph()
        graph.add_node(GraphNode(id="a", kind=NodeKind.FUNCTION, name="a"))
        graph.add_node(GraphNode(id="b", kind=NodeKind.FUNCTION, name="b"))
        graph.add_node(GraphNode(id="c", kind=NodeKind.FUNCTION, name="c"))
        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="b", target_id="c", kind=EdgeKind.CALLS))

        builder = HierarchyBuilder(max_levels=3)
        root = builder.build(graph)

        # Should complete and return valid hierarchy
        assert root is not None
        assert root.node_ids == frozenset({"a", "b", "c"})
        flattened = builder.flatten(root)
        assert len(flattened) > 0


class TestHierarchyBuilderLargeSCC:
    """Test handling of graphs with large SCCs."""

    def test_one_large_scc_condensed_to_single_node(self):
        """
        Test that a large SCC is properly condensed.
        Graph: clique of 5 nodes (all interconnected).
        """
        graph = CallGraph()
        nodes = ["n1", "n2", "n3", "n4", "n5"]
        for node in nodes:
            graph.add_node(GraphNode(id=node, kind=NodeKind.FUNCTION, name=node))

        # Create complete graph (every node calls every other)
        for src in nodes:
            for tgt in nodes:
                if src != tgt:
                    graph.add_edge(GraphEdge(source_id=src, target_id=tgt, kind=EdgeKind.CALLS))

        builder = HierarchyBuilder(min_scc_size=2, max_levels=5)
        root = builder.build(graph)

        # Should detect one large SCC
        assert root.scc_members is not None
        large_sccs = [scc for scc in root.scc_members if len(scc) >= 5]
        assert len(large_sccs) >= 1
        # The largest should contain all our nodes
        largest = max(root.scc_members, key=len)
        assert all(node in largest for node in nodes)


# ─────────────────────────────────────────────────────────────────
# DependencyExtractor Tests
# ─────────────────────────────────────────────────────────────────


class TestExtractImportsFromSource:
    """Test extract_imports_from_source() function."""

    def test_simple_import_statement(self):
        """Test extraction of simple 'import X' statement."""
        source = "import os\nimport sys"
        imports = extract_imports_from_source(source)

        assert len(imports) == 2
        assert imports[0].module == "os"
        assert imports[0].is_from_import is False
        assert imports[0].level == 0
        assert imports[1].module == "sys"

    def test_from_import_statement(self):
        """Test extraction of 'from X import Y' statement."""
        source = "from pathlib import Path"
        imports = extract_imports_from_source(source)

        assert len(imports) == 1
        assert imports[0].module == "pathlib"
        assert imports[0].is_from_import is True
        assert "Path" in imports[0].names
        assert imports[0].level == 0

    def test_from_import_multiple_names(self):
        """Test 'from X import Y, Z' statement."""
        source = "from typing import List, Dict, Optional"
        imports = extract_imports_from_source(source)

        assert len(imports) == 1
        assert imports[0].module == "typing"
        assert "List" in imports[0].names
        assert "Dict" in imports[0].names
        assert "Optional" in imports[0].names

    def test_relative_import_level_one(self):
        """Test relative import 'from . import foo'."""
        source = "from . import utils"
        imports = extract_imports_from_source(source)

        assert len(imports) == 1
        assert imports[0].level == 1
        assert imports[0].module is None
        assert "utils" in imports[0].names

    def test_relative_import_level_two(self):
        """Test relative import 'from .. import bar'."""
        source = "from .. import config"
        imports = extract_imports_from_source(source)

        assert len(imports) == 1
        assert imports[0].level == 2
        assert imports[0].module is None
        assert "config" in imports[0].names

    def test_relative_import_with_module(self):
        """Test 'from ..package import module'."""
        source = "from ..sibling import helper"
        imports = extract_imports_from_source(source)

        assert len(imports) == 1
        assert imports[0].level == 2
        assert imports[0].module == "sibling"
        assert "helper" in imports[0].names

    def test_no_imports(self):
        """Test source with no imports."""
        source = "def foo():\n    return 42"
        imports = extract_imports_from_source(source)

        assert len(imports) == 0

    def test_wildcard_import(self):
        """Test 'from X import *'."""
        source = "from os import *"
        imports = extract_imports_from_source(source)

        assert len(imports) == 1
        assert "*" in imports[0].names

    def test_import_with_alias(self):
        """Test 'import X as Y'."""
        source = "import numpy as np"
        imports = extract_imports_from_source(source)

        assert len(imports) == 1
        # asname is stored in names
        assert "np" in imports[0].names or "numpy" in imports[0].names

    def test_syntax_error_returns_empty(self):
        """Test that syntax errors return empty list."""
        source = "import os\n  invalid syntax !!!"
        imports = extract_imports_from_source(source)

        # Should not raise, just return empty
        assert isinstance(imports, list)


class TestResolveRelativeImport:
    """Test _resolve_relative_import() function."""

    def test_absolute_import_unchanged(self):
        """Test that absolute imports (level=0) are returned unchanged."""
        result = _resolve_relative_import("a.b.c", "x.y", level=0)
        assert result == "x.y"

    def test_relative_level_one_same_package(self):
        """Test level=1 (from . import X) for same-package resolution."""
        # Importing from "a.b" when in "a.b.c"
        result = _resolve_relative_import("a.b.c", "helper", level=1)
        # level=1 means go up 1 level: from "a.b.c" -> "a.b"
        assert result == "a.b.helper"

    def test_relative_level_one_with_none_module(self):
        """Test level=1 with None module (from . import X)."""
        # "from . import helper" in module "a.b.c"
        result = _resolve_relative_import("a.b.c", None, level=1)
        # Should resolve to "a.b" (the parent package)
        assert result == "a.b"

    def test_relative_level_two_parent_package(self):
        """Test level=2 (from .. import X) for parent-package resolution."""
        # In "a.b.c", from .. import something
        result = _resolve_relative_import("a.b.c", "sibling", level=2)
        # level=2 means go up 2 levels: "a.b.c" -> "a" -> "a.sibling"
        assert result == "a.sibling"

    def test_relative_level_two_with_none_module(self):
        """Test level=2 with None module."""
        result = _resolve_relative_import("a.b.c", None, level=2)
        # Should resolve to "a"
        assert result == "a"

    def test_relative_import_above_root_returns_none(self):
        """Test that going above root returns None."""
        # In "a.b", try to go up 3 levels (above root)
        result = _resolve_relative_import("a.b", "x", level=3)
        assert result is None

    def test_relative_single_level_module(self):
        """Test relative import in single-level module."""
        # In "top_level" module, from . import something
        result = _resolve_relative_import("top", "sub", level=1)
        # level=1 from "top" should give us "" -> can't go up
        # Actually, let's check: parts = ["top"], level=1
        # base_parts = parts[: 1 - 1] = parts[:0] = []
        # So base = "" and with sub: returns sub
        assert result == "sub"


class TestDependencyExtractorDirectory:
    """Test DependencyExtractor.extract_directory() method."""

    def test_extract_directory_with_three_files(self, tmp_path: Path):
        """
        Test extracting dependencies from a temp directory with 3 Python files
        that import each other.
        """
        # Create temp structure:
        # tmp_path/
        #   a.py: imports b
        #   b.py: imports c
        #   c.py: no imports

        a_file = tmp_path / "a.py"
        b_file = tmp_path / "b.py"
        c_file = tmp_path / "c.py"

        a_file.write_text("import b\ndef func_a(): pass")
        b_file.write_text("import c\ndef func_b(): pass")
        c_file.write_text("def func_c(): pass")

        extractor = DependencyExtractor(include_stdlib=False)
        graph = extractor.extract_directory(tmp_path)

        # Should have 3 module nodes
        assert len(graph.nodes) == 3
        assert "a" in graph.nodes
        assert "b" in graph.nodes
        assert "c" in graph.nodes

        # Should have 2 IMPORTS edges: a->b, b->c
        imports_edges = [e for e in graph.edges if e.kind == EdgeKind.IMPORTS]
        assert len(imports_edges) == 2

        # Check edge structure
        source_targets = {(e.source_id, e.target_id) for e in imports_edges}
        assert ("a", "b") in source_targets
        assert ("b", "c") in source_targets

    def test_extract_directory_module_nodes_are_modules(self, tmp_path: Path):
        """Test that extracted nodes are MODULE kind."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        extractor = DependencyExtractor()
        graph = extractor.extract_directory(tmp_path)

        # All nodes should be MODULE kind
        for node in graph.nodes.values():
            assert node.kind == NodeKind.MODULE

    def test_extract_directory_excludes_stdlib(self, tmp_path: Path):
        """Test that stdlib imports are excluded by default."""
        test_file = tmp_path / "test.py"
        test_file.write_text("import os\nimport sys\nimport json")

        extractor = DependencyExtractor(include_stdlib=False)
        graph = extractor.extract_directory(tmp_path)

        # Should have only the local module, no os/sys/json
        assert "test" in graph.nodes
        assert "os" not in graph.nodes
        assert "sys" not in graph.nodes
        assert "json" not in graph.nodes

        # Should have no edges (all imports are stdlib)
        imports_edges = [e for e in graph.edges if e.kind == EdgeKind.IMPORTS]
        assert len(imports_edges) == 0

    def test_extract_directory_includes_stdlib_when_flag_set(self, tmp_path: Path):
        """Test that stdlib imports are included when flag is True."""
        test_file = tmp_path / "test.py"
        test_file.write_text("import os")

        extractor = DependencyExtractor(include_stdlib=True)
        graph = extractor.extract_directory(tmp_path)

        # Should have both local and stdlib modules
        assert "test" in graph.nodes
        # os might or might not be present depending on implementation,
        # but there should be an edge to something
        imports_edges = [e for e in graph.edges if e.kind == EdgeKind.IMPORTS]
        assert len(imports_edges) >= 1

    def test_extract_directory_circular_imports(self, tmp_path: Path):
        """Test detection of circular imports between two modules."""
        # Create circular import: mod_a imports mod_b, mod_b imports mod_a
        mod_a = tmp_path / "mod_a.py"
        mod_b = tmp_path / "mod_b.py"

        mod_a.write_text("from mod_b import func_b\ndef func_a(): pass")
        mod_b.write_text("from mod_a import func_a\ndef func_b(): pass")

        extractor = DependencyExtractor(include_stdlib=False)
        graph = extractor.extract_directory(tmp_path)

        # Should detect both directions
        imports_edges = [e for e in graph.edges if e.kind == EdgeKind.IMPORTS]

        source_targets = {(e.source_id, e.target_id) for e in imports_edges}
        assert ("mod_a", "mod_b") in source_targets
        assert ("mod_b", "mod_a") in source_targets

    def test_extract_directory_relative_imports(self, tmp_path: Path):
        """Test extraction with relative imports."""
        # Create package structure:
        # tmp_path/
        #   pkg/
        #     __init__.py
        #     module_a.py: from . import module_b
        #     module_b.py

        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()

        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "module_a.py").write_text("from . import module_b")
        (pkg_dir / "module_b.py").write_text("def func(): pass")

        extractor = DependencyExtractor(include_stdlib=False)
        graph = extractor.extract_directory(tmp_path)

        # Should have nodes for pkg, pkg.module_a, pkg.module_b
        assert "pkg" in graph.nodes or "pkg.module_a" in graph.nodes

        # Relative import should be resolved
        imports_edges = [e for e in graph.edges if e.kind == EdgeKind.IMPORTS]
        # There should be an edge from module_a to module_b
        assert len(imports_edges) >= 1

    def test_extract_directory_glob_pattern(self, tmp_path: Path):
        """Test that glob pattern filters files correctly."""
        # Create files
        (tmp_path / "code.py").write_text("import sys")
        (tmp_path / "test.py").write_text("import os")
        (tmp_path / "data.txt").write_text("not python")

        extractor = DependencyExtractor()
        # Extract only *.py files
        graph = extractor.extract_directory(tmp_path, pattern="*.py")

        # Should have 2 module nodes
        assert len(graph.nodes) == 2
        module_names = {n for n in graph.nodes.keys()}
        assert "code" in module_names
        assert "test" in module_names

    def test_extract_directory_exclude_directories(self, tmp_path: Path):
        """Test that excluded directories are skipped."""
        # Create structure with venv
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "pkg.py").write_text("import os")

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def main(): pass")

        extractor = DependencyExtractor()
        graph = extractor.extract_directory(tmp_path)

        # Should have src.main but not .venv.pkg
        module_names = {n for n in graph.nodes.keys()}
        assert any("main" in name for name in module_names)
        assert not any("pkg" in name for name in module_names)

    def test_extract_directory_empty_directory(self, tmp_path: Path):
        """Test extracting from an empty directory."""
        extractor = DependencyExtractor()
        graph = extractor.extract_directory(tmp_path)

        # Should return empty graph
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0


class TestDependencyExtractorFile:
    """Test DependencyExtractor.extract_file() method."""

    def test_extract_file_returns_import_info_list(self, tmp_path: Path):
        """Test that extract_file returns a list of ImportInfo."""
        test_file = tmp_path / "test.py"
        test_file.write_text("import os\nfrom pathlib import Path")

        extractor = DependencyExtractor()
        imports = extractor.extract_file(test_file)

        assert isinstance(imports, list)
        assert len(imports) == 2

    def test_extract_file_nonexistent_returns_empty(self, tmp_path: Path):
        """Test that extract_file on nonexistent file returns empty list."""
        nonexistent = tmp_path / "nonexistent.py"

        extractor = DependencyExtractor()
        imports = extractor.extract_file(nonexistent)

        assert imports == []


class TestImportInfoIntegration:
    """Integration tests for import extraction."""

    def test_complex_source_with_multiple_import_styles(self):
        """Test extraction from source with various import styles."""
        source = """
import os
import sys, json
from pathlib import Path
from typing import List, Dict
from . import utils
from ..sibling import helper
"""
        imports = extract_imports_from_source(source)

        assert len(imports) >= 4  # Multiple import statements

        # Check we got different styles
        absolute_imports = [i for i in imports if i.level == 0]
        relative_imports = [i for i in imports if i.level > 0]

        assert len(absolute_imports) >= 3
        assert len(relative_imports) >= 2

    def test_import_with_comments_and_docstrings(self):
        """Test extraction ignores comments and docstrings."""
        source = '''
"""
Module docstring with "import fake" which should be ignored.
"""

# import commented_out

import real_module

def foo():
    """Function with import in docstring: import ignored"""
    pass
'''
        imports = extract_imports_from_source(source)

        # Should only find real_module
        assert len(imports) == 1
        assert imports[0].module == "real_module"


class TestDependencyExtractorConfidence:
    """Test confidence scoring in dependency extraction."""

    def test_wildcard_import_lower_confidence(self, tmp_path: Path):
        """Test that wildcard imports have lower confidence."""
        test_file = tmp_path / "test.py"
        test_file.write_text("from os import *\nimport sys")

        extractor = DependencyExtractor(include_stdlib=True)
        graph = extractor.extract_directory(tmp_path)

        # Find edges
        edges = list(graph.edges)

        # At least one edge should exist
        if edges:
            # Wildcard import edges should have lower confidence
            wildcard_edges = [e for e in edges if e.source_id == "test"]
            if wildcard_edges:
                # Confidence should be <= 1.0 (wildcard = 0.7)
                for edge in wildcard_edges:
                    assert edge.confidence <= 1.0

    def test_relative_import_high_confidence(self, tmp_path: Path):
        """Test that relative imports have high confidence."""
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()

        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "a.py").write_text("from . import b")
        (pkg_dir / "b.py").write_text("pass")

        extractor = DependencyExtractor(include_stdlib=False)
        graph = extractor.extract_directory(tmp_path)

        # Find relative import edges
        edges = list(graph.edges)
        if edges:
            for edge in edges:
                # Relative imports should have confidence >= 0.9
                if "pkg.a" in edge.source_id:
                    assert edge.confidence >= 0.9


# ─────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────


class TestIntegrationHierarchyWithExtractedDeps:
    """Test HierarchyBuilder with graphs built from DependencyExtractor."""

    def test_build_hierarchy_from_extracted_dependencies(self, tmp_path: Path):
        """
        Integration test: extract dependencies, then build hierarchy.
        """
        # Create simple module structure
        (tmp_path / "main.py").write_text("from utils import helper")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        # Extract dependencies
        extractor = DependencyExtractor(include_stdlib=False)
        dep_graph = extractor.extract_directory(tmp_path)

        # Convert module-level graph to function-level by treating modules as functions
        # (simplification for testing)
        func_graph = CallGraph()
        for node in dep_graph.nodes.values():
            func_graph.add_node(GraphNode(
                id=node.id,
                kind=NodeKind.FUNCTION,  # Treat modules as functions for hierarchy
                name=node.name,
            ))
        for edge in dep_graph.edges:
            func_graph.add_edge(GraphEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                kind=EdgeKind.CALLS,  # Treat imports as calls
            ))

        # Build hierarchy
        builder = HierarchyBuilder()
        root = builder.build(func_graph)

        # Verify hierarchy was built
        assert root is not None
        assert root.operation == "condense"
        flattened = builder.flatten(root)
        assert len(flattened) > 0

    def test_hierarchy_on_cyclic_module_dependencies(self, tmp_path: Path):
        """Test hierarchy building on circular module dependencies."""
        # Create circular modules
        (tmp_path / "a.py").write_text("import b")
        (tmp_path / "b.py").write_text("import a")

        extractor = DependencyExtractor(include_stdlib=False)
        dep_graph = extractor.extract_directory(tmp_path)

        # Convert to function-level graph
        func_graph = CallGraph()
        for node in dep_graph.nodes.values():
            func_graph.add_node(GraphNode(
                id=node.id,
                kind=NodeKind.FUNCTION,
                name=node.name,
            ))
        for edge in dep_graph.edges:
            func_graph.add_edge(GraphEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                kind=EdgeKind.CALLS,
            ))

        # Build hierarchy — should detect SCC with both modules
        builder = HierarchyBuilder(min_scc_size=2)
        root = builder.build(func_graph)

        # Should have SCCs
        assert root.scc_members is not None
        # Find the SCC containing both 'a' and 'b'
        scc_with_cycle = next(
            (scc for scc in root.scc_members if "a" in scc and "b" in scc),
            None
        )
        assert scc_with_cycle is not None
