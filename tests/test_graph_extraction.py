"""
Tests for call graph extraction.

Tests the AST-based extractor with various Python code patterns.
"""

import sys
from pathlib import Path
from textwrap import dedent

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph import (
    ASTExtractor,
    CallGraph,
    EdgeKind,
    FunctionSignature,
    GraphEdge,
    GraphNode,
    NodeKind,
    get_extractor,
)


class TestASTExtractor:
    """Tests for the AST-based call graph extractor."""

    @pytest.fixture
    def extractor(self):
        return ASTExtractor()

    def test_extract_simple_function(self, extractor):
        """Test extracting a simple function definition."""
        source = dedent("""
            def hello():
                print("Hello, world!")
        """)

        graph = extractor.extract_module(source, "test_module")

        # Should have module + function
        assert len(graph.nodes) == 2

        # Check function node
        func = graph.get_node("test_module.hello")
        assert func is not None
        assert func.kind == NodeKind.FUNCTION
        assert func.name == "hello"
        assert func.signature is not None
        assert func.signature.params == ()

    def test_extract_function_with_params(self, extractor):
        """Test extracting function with parameters."""
        source = dedent("""
            def greet(name: str, times: int = 1) -> str:
                return name * times
        """)

        graph = extractor.extract_module(source, "test_module")
        func = graph.get_node("test_module.greet")

        assert func is not None
        assert func.signature.params == ("name", "times")
        assert func.signature.return_type == "str"

    def test_extract_async_function(self, extractor):
        """Test extracting async function."""
        source = dedent("""
            async def fetch_data(url: str):
                pass
        """)

        graph = extractor.extract_module(source, "test_module")
        func = graph.get_node("test_module.fetch_data")

        assert func is not None
        assert func.signature.is_async is True

    def test_extract_class_with_methods(self, extractor):
        """Test extracting class with methods."""
        source = dedent("""
            class Calculator:
                def __init__(self, value: int):
                    self.value = value

                def add(self, x: int) -> int:
                    return self.value + x

                def multiply(self, x: int) -> int:
                    return self.value * x
        """)

        graph = extractor.extract_module(source, "test_module")

        # Check class node
        cls = graph.get_node("test_module.Calculator")
        assert cls is not None
        assert cls.kind == NodeKind.CLASS

        # Check method nodes
        init = graph.get_node("test_module.Calculator.__init__")
        assert init is not None
        assert init.kind == NodeKind.METHOD

        add = graph.get_node("test_module.Calculator.add")
        assert add is not None
        assert add.kind == NodeKind.METHOD

    def test_extract_function_calls(self, extractor):
        """Test extracting function call edges."""
        source = dedent("""
            def helper():
                return 42

            def main():
                x = helper()
                return x
        """)

        graph = extractor.extract_module(source, "test_module")

        # Check call edge
        callees = graph.get_callees("test_module.main")
        assert "test_module.helper" in callees

    def test_extract_method_calls(self, extractor):
        """Test extracting method call edges."""
        source = dedent("""
            class Foo:
                def bar(self):
                    return 1

                def baz(self):
                    return self.bar() + 1
        """)

        graph = extractor.extract_module(source, "test_module")

        # baz calls bar via self
        callees = graph.get_callees("test_module.Foo.baz")
        # Due to dynamic nature, this might be "self.bar"
        assert any("bar" in c for c in callees)

    def test_extract_nested_functions(self, extractor):
        """Test extracting nested function definitions."""
        source = dedent("""
            def outer():
                def inner():
                    return 1
                return inner()
        """)

        graph = extractor.extract_module(source, "test_module")

        outer = graph.get_node("test_module.outer")
        inner = graph.get_node("test_module.outer.inner")

        assert outer is not None
        assert inner is not None
        assert inner.kind == NodeKind.FUNCTION

        # outer calls inner
        callees = graph.get_callees("test_module.outer")
        assert "test_module.outer.inner" in callees

    def test_extract_lambda(self, extractor):
        """Test extracting lambda expressions."""
        source = dedent("""
            def main():
                f = lambda x: x + 1
                return f(10)
        """)

        graph = extractor.extract_module(source, "test_module")

        # Should find a lambda node
        lambdas = [n for n in graph.nodes.values() if n.kind == NodeKind.LAMBDA]
        assert len(lambdas) == 1

    def test_extract_inheritance(self, extractor):
        """Test extracting class inheritance edges."""
        source = dedent("""
            class Base:
                pass

            class Derived(Base):
                pass
        """)

        graph = extractor.extract_module(source, "test_module")

        # Check inheritance edge
        inherits_edges = list(graph.get_edges_from("test_module.Derived", EdgeKind.INHERITS))
        assert len(inherits_edges) == 1
        assert inherits_edges[0].target_id == "Base"

    def test_extract_defines_edges(self, extractor):
        """Test that defines edges are created."""
        source = dedent("""
            class MyClass:
                def method(self):
                    pass
        """)

        graph = extractor.extract_module(source, "test_module")

        # Module defines class
        defines = list(graph.get_edges_from("test_module", EdgeKind.DEFINES))
        targets = {e.target_id for e in defines}
        assert "test_module.MyClass" in targets

        # Class defines method
        class_defines = list(graph.get_edges_from("test_module.MyClass", EdgeKind.DEFINES))
        class_targets = {e.target_id for e in class_defines}
        assert "test_module.MyClass.method" in class_targets

    def test_builtin_calls(self, extractor):
        """Test that builtin calls are tracked."""
        source = dedent("""
            def main():
                x = len([1, 2, 3])
                print(x)
        """)

        graph = extractor.extract_module(source, "test_module")

        callees = graph.get_callees("test_module.main")
        assert "builtins.len" in callees
        assert "builtins.print" in callees


class TestCallGraphOperations:
    """Tests for CallGraph methods."""

    def test_reachable_from(self):
        """Test reachability query."""
        graph = CallGraph()

        # Create chain: a -> b -> c -> d
        for name in ["a", "b", "c", "d"]:
            graph.add_node(GraphNode(id=name, kind=NodeKind.FUNCTION, name=name))

        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="b", target_id="c", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="c", target_id="d", kind=EdgeKind.CALLS))

        reachable = graph.reachable_from("a")
        assert reachable == {"b", "c", "d"}

        reachable_2 = graph.reachable_from("a", max_depth=2)
        assert reachable_2 == {"b", "c"}

    def test_reaches(self):
        """Test reverse reachability query."""
        graph = CallGraph()

        # Create chain: a -> b -> c
        for name in ["a", "b", "c"]:
            graph.add_node(GraphNode(id=name, kind=NodeKind.FUNCTION, name=name))

        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="b", target_id="c", kind=EdgeKind.CALLS))

        reaches = graph.reaches("c")
        assert reaches == {"a", "b"}

    def test_strongly_connected_components(self):
        """Test SCC detection."""
        graph = CallGraph()

        # Create cycle: a -> b -> c -> a
        for name in ["a", "b", "c", "d"]:
            graph.add_node(GraphNode(id=name, kind=NodeKind.FUNCTION, name=name))

        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="b", target_id="c", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="c", target_id="a", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="a", target_id="d", kind=EdgeKind.CALLS))

        sccs = graph.strongly_connected_components()

        # Should have 2 SCCs: {a, b, c} and {d}
        assert len(sccs) == 2

        # Find the cycle SCC
        cycle_scc = next(scc for scc in sccs if len(scc) == 3)
        assert cycle_scc == frozenset({"a", "b", "c"})

    def test_condensation(self):
        """Test condensation graph."""
        graph = CallGraph()

        # Create: (a <-> b) -> c
        for name in ["a", "b", "c"]:
            graph.add_node(GraphNode(id=name, kind=NodeKind.FUNCTION, name=name))

        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="b", target_id="a", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="a", target_id="c", kind=EdgeKind.CALLS))

        condensed = graph.condensation()

        # Should have 2 nodes (SCCs)
        assert len(condensed.nodes) == 2

        # Should have 1 edge between SCCs
        assert len(condensed.edges) == 1

    def test_topological_sort_dag(self):
        """Test topological sort on DAG."""
        graph = CallGraph()

        # Create DAG: a -> b -> d, a -> c -> d
        for name in ["a", "b", "c", "d"]:
            graph.add_node(GraphNode(id=name, kind=NodeKind.FUNCTION, name=name))

        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="a", target_id="c", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="b", target_id="d", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="c", target_id="d", kind=EdgeKind.CALLS))

        order = graph.topological_sort()

        # a must come before b, c; b and c must come before d
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_topological_sort_cycle_raises(self):
        """Test that topological sort raises on cycle."""
        graph = CallGraph()

        for name in ["a", "b"]:
            graph.add_node(GraphNode(id=name, kind=NodeKind.FUNCTION, name=name))

        graph.add_edge(GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS))
        graph.add_edge(GraphEdge(source_id="b", target_id="a", kind=EdgeKind.CALLS))

        with pytest.raises(ValueError, match="cycles"):
            graph.topological_sort()

    def test_to_dot(self):
        """Test DOT export."""
        graph = CallGraph()

        graph.add_node(GraphNode(id="main", kind=NodeKind.FUNCTION, name="main"))
        graph.add_node(GraphNode(id="helper", kind=NodeKind.FUNCTION, name="helper"))
        graph.add_edge(GraphEdge(source_id="main", target_id="helper", kind=EdgeKind.CALLS))

        dot = graph.to_dot("Test Graph")

        assert 'digraph "Test Graph"' in dot
        assert '"main"' in dot
        assert '"helper"' in dot
        assert '"main" -> "helper"' in dot

    def test_serialization_roundtrip(self):
        """Test JSON serialization and deserialization."""
        graph = CallGraph()

        graph.add_node(
            GraphNode(
                id="test.func",
                kind=NodeKind.FUNCTION,
                name="func",
                signature=FunctionSignature(name="func", params=("x", "y")),
            )
        )
        graph.add_edge(
            GraphEdge(
                source_id="test.func",
                target_id="builtins.print",
                kind=EdgeKind.CALLS,
            )
        )

        # Round-trip
        data = graph.to_dict()
        restored = CallGraph.from_dict(data)

        assert len(restored.nodes) == len(graph.nodes)
        assert len(restored.edges) == len(graph.edges)

        func = restored.get_node("test.func")
        assert func is not None
        assert func.signature.params == ("x", "y")


class TestExtractorFactory:
    """Tests for the extractor factory function."""

    def test_get_ast_extractor(self):
        """Test getting AST extractor explicitly."""
        extractor = get_extractor(backend="ast")
        assert extractor.backend_name == "ast"

    def test_get_auto_extractor(self):
        """Test auto-selection (should fall back to AST)."""
        extractor = get_extractor(backend="auto")
        # Without LPython installed, should get AST
        assert extractor.backend_name in ("ast", "asr")

    def test_invalid_backend_raises(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_extractor(backend="invalid")


class TestComplexPatterns:
    """Tests for complex code patterns."""

    @pytest.fixture
    def extractor(self):
        return ASTExtractor()

    def test_decorator_functions(self, extractor):
        """Test functions with decorators."""
        source = dedent("""
            def decorator(func):
                def wrapper(*args):
                    return func(*args)
                return wrapper

            @decorator
            def decorated():
                pass
        """)

        graph = extractor.extract_module(source, "test_module")

        func = graph.get_node("test_module.decorated")
        assert func is not None
        assert "decorator" in func.signature.decorators

    def test_generator_function(self, extractor):
        """Test generator function detection."""
        source = dedent("""
            def counter(n: int):
                for i in range(n):
                    yield i
        """)

        graph = extractor.extract_module(source, "test_module")

        func = graph.get_node("test_module.counter")
        assert func is not None
        assert func.signature.is_generator is True

    def test_classmethod_staticmethod(self, extractor):
        """Test classmethod and staticmethod."""
        source = dedent("""
            class MyClass:
                @classmethod
                def from_string(cls, s: str):
                    return cls()

                @staticmethod
                def utility():
                    return 42
        """)

        graph = extractor.extract_module(source, "test_module")

        from_string = graph.get_node("test_module.MyClass.from_string")
        assert from_string is not None
        assert "classmethod" in from_string.signature.decorators

        utility = graph.get_node("test_module.MyClass.utility")
        assert utility is not None
        assert "staticmethod" in utility.signature.decorators

    def test_property_methods(self, extractor):
        """Test property methods."""
        source = dedent("""
            class Circle:
                def __init__(self, radius: float):
                    self._radius = radius

                @property
                def area(self) -> float:
                    return 3.14159 * self._radius ** 2
        """)

        graph = extractor.extract_module(source, "test_module")

        area = graph.get_node("test_module.Circle.area")
        assert area is not None
        assert "property" in area.signature.decorators


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
