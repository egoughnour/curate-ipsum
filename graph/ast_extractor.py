"""
Python AST-based call graph extractor.

Uses Python's built-in ast module to extract function definitions,
calls, and their relationships. This is always available without
external dependencies.

The extractor performs two passes:
1. Definition pass: Collect all function/class/method definitions
2. Call pass: Collect all calls and resolve targets

For unresolved calls (e.g., method calls on unknown objects), we use
heuristics and mark edges with lower confidence.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from .extractor import CallGraphExtractor, ParseError
from .models import (
    CallGraph,
    EdgeKind,
    FunctionSignature,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
)


def _get_end_lineno(node: ast.AST, default: int = 1) -> int:
    """Safely get end_lineno from AST node."""
    return getattr(node, "end_lineno", None) or getattr(node, "lineno", default) or default


class ScopeTracker:
    """
    Tracks the current scope during AST traversal.

    Maintains a stack of scope names to build fully qualified names.
    """

    def __init__(self, module_name: str):
        self._stack: list[tuple[str, NodeKind]] = [(module_name, NodeKind.MODULE)]

    def push(self, name: str, kind: NodeKind) -> None:
        """Enter a new scope."""
        self._stack.append((name, kind))

    def pop(self) -> tuple[str, NodeKind]:
        """Exit current scope."""
        return self._stack.pop()

    @property
    def current_fqn(self) -> str:
        """Fully qualified name of current scope."""
        return ".".join(name for name, _ in self._stack)

    @property
    def current_kind(self) -> NodeKind:
        """Kind of current scope."""
        return self._stack[-1][1]

    def fqn_for(self, name: str) -> str:
        """Build FQN for a name in current scope."""
        return f"{self.current_fqn}.{name}"

    @property
    def depth(self) -> int:
        """Current nesting depth."""
        return len(self._stack)


class DefinitionVisitor(ast.NodeVisitor):
    """
    First pass: collect all definitions (functions, classes, methods).

    Builds a symbol table mapping names to their nodes.
    """

    def __init__(
        self,
        file_path: str,
        module_name: str,
        include_lambdas: bool = True,
        include_comprehensions: bool = False,
    ):
        self.file_path = file_path
        self.scope = ScopeTracker(module_name)
        self.include_lambdas = include_lambdas
        self.include_comprehensions = include_comprehensions

        self.graph = CallGraph()
        self.symbol_table: dict[str, str] = {}  # local name -> FQN
        self._lambda_counter = 0
        self._comp_counter = 0

    def visit_Module(self, node: ast.Module) -> None:
        """Visit module node."""
        # Add module node
        module_node = GraphNode(
            id=self.scope.current_fqn,
            kind=NodeKind.MODULE,
            name=self.scope.current_fqn,
            location=SourceLocation(
                file=self.file_path,
                line_start=1,
                line_end=_get_end_lineno(node, 1),
            ),
        )
        self.graph.add_node(module_node)

        # Visit children
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        fqn = self.scope.fqn_for(node.name)

        class_node = GraphNode(
            id=fqn,
            kind=NodeKind.CLASS,
            name=node.name,
            location=SourceLocation(
                file=self.file_path,
                line_start=node.lineno,
                line_end=_get_end_lineno(node, node.lineno),
            ),
            docstring=ast.get_docstring(node),
        )
        self.graph.add_node(class_node)

        # Add "defines" edge from parent scope
        self.graph.add_edge(
            GraphEdge(
                source_id=self.scope.current_fqn,
                target_id=fqn,
                kind=EdgeKind.DEFINES,
                location=SourceLocation(
                    file=self.file_path,
                    line_start=node.lineno,
                    line_end=node.lineno,
                ),
            )
        )

        # Add inheritance edges
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                self.graph.add_edge(
                    GraphEdge(
                        source_id=fqn,
                        target_id=base_name,  # May be unresolved
                        kind=EdgeKind.INHERITS,
                        confidence=0.8 if "." not in base_name else 0.5,
                    )
                )

        # Visit class body
        self.scope.push(node.name, NodeKind.CLASS)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function/method definition."""
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function/method definition."""
        self._visit_function(node, is_async=True)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool,
    ) -> None:
        """Common handler for function definitions."""
        fqn = self.scope.fqn_for(node.name)

        # Determine if this is a method
        is_method = self.scope.current_kind == NodeKind.CLASS
        kind = NodeKind.METHOD if is_method else NodeKind.FUNCTION

        # Extract parameters
        params = self._extract_params(node.args)

        # Extract return type annotation
        return_type = None
        if node.returns:
            return_type = self._get_annotation_str(node.returns)

        # Extract decorators
        decorators = tuple(self._get_name(d) or "<unknown>" for d in node.decorator_list)

        # Check if generator
        is_generator = self._contains_yield(node)

        signature = FunctionSignature(
            name=node.name,
            params=params,
            return_type=return_type,
            decorators=decorators,
            is_async=is_async,
            is_generator=is_generator,
        )

        func_node = GraphNode(
            id=fqn,
            kind=kind,
            name=node.name,
            location=SourceLocation(
                file=self.file_path,
                line_start=node.lineno,
                line_end=_get_end_lineno(node, node.lineno),
            ),
            signature=signature,
            docstring=ast.get_docstring(node),
        )
        self.graph.add_node(func_node)

        # Add "defines" edge from parent scope
        self.graph.add_edge(
            GraphEdge(
                source_id=self.scope.current_fqn,
                target_id=fqn,
                kind=EdgeKind.DEFINES,
                location=SourceLocation(
                    file=self.file_path,
                    line_start=node.lineno,
                    line_end=node.lineno,
                ),
            )
        )

        # Store in symbol table for call resolution
        self.symbol_table[node.name] = fqn

        # Visit function body
        self.scope.push(node.name, kind)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Visit lambda expression."""
        if not self.include_lambdas:
            return

        self._lambda_counter += 1
        name = f"<lambda_{self._lambda_counter}>"
        fqn = self.scope.fqn_for(name)

        params = self._extract_params(node.args)
        signature = FunctionSignature(name=name, params=params)

        lambda_node = GraphNode(
            id=fqn,
            kind=NodeKind.LAMBDA,
            name=name,
            location=SourceLocation(
                file=self.file_path,
                line_start=node.lineno,
                line_end=_get_end_lineno(node, node.lineno),
            ),
            signature=signature,
        )
        self.graph.add_node(lambda_node)

        # Add "defines" edge
        self.graph.add_edge(
            GraphEdge(
                source_id=self.scope.current_fqn,
                target_id=fqn,
                kind=EdgeKind.DEFINES,
            )
        )

        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, "listcomp")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, "setcomp")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, "dictcomp")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, "genexp")

    def _visit_comprehension(self, node: ast.AST, kind_str: str) -> None:
        """Visit comprehension expression."""
        if not self.include_comprehensions:
            return

        self._comp_counter += 1
        name = f"<{kind_str}_{self._comp_counter}>"
        fqn = self.scope.fqn_for(name)

        comp_node = GraphNode(
            id=fqn,
            kind=NodeKind.COMPREHENSION,
            name=name,
            location=SourceLocation(
                file=self.file_path,
                line_start=node.lineno,
                line_end=_get_end_lineno(node, node.lineno),
            ),
        )
        self.graph.add_node(comp_node)

        self.graph.add_edge(
            GraphEdge(
                source_id=self.scope.current_fqn,
                target_id=fqn,
                kind=EdgeKind.DEFINES,
            )
        )

        self.generic_visit(node)

    def _extract_params(self, args: ast.arguments) -> tuple[str, ...]:
        """Extract parameter names from function arguments."""
        params: list[str] = []

        # Positional-only (Python 3.8+)
        for arg in args.posonlyargs:
            params.append(arg.arg)

        # Regular positional/keyword
        for arg in args.args:
            params.append(arg.arg)

        # *args
        if args.vararg:
            params.append(f"*{args.vararg.arg}")

        # Keyword-only
        for arg in args.kwonlyargs:
            params.append(arg.arg)

        # **kwargs
        if args.kwarg:
            params.append(f"**{args.kwarg.arg}")

        return tuple(params)

    def _get_name(self, node: ast.AST) -> str | None:
        """Extract name from various AST node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        return None

    def _get_annotation_str(self, node: ast.AST) -> str:
        """Convert annotation AST node to string."""
        try:
            return ast.unparse(node)
        except Exception:
            return "<unknown>"

    def _contains_yield(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function contains yield/yield from."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False


class CallVisitor(ast.NodeVisitor):
    """
    Second pass: collect all function/method calls.

    Uses the symbol table from the definition pass to resolve call targets.
    """

    def __init__(
        self,
        file_path: str,
        module_name: str,
        graph: CallGraph,
        symbol_table: dict[str, str],
        include_dynamic_calls: bool = True,
    ):
        self.file_path = file_path
        self.scope = ScopeTracker(module_name)
        self.graph = graph
        self.symbol_table = symbol_table
        self.include_dynamic_calls = include_dynamic_calls

        # Track assignments for better resolution
        self.local_assignments: dict[str, str] = {}

    def visit_Module(self, node: ast.Module) -> None:
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.push(node.name, NodeKind.CLASS)
        old_locals = self.local_assignments.copy()
        self.local_assignments.clear()
        self.generic_visit(node)
        self.local_assignments = old_locals
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_body(node)

    def _visit_function_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Visit function body to find calls."""
        kind = NodeKind.METHOD if self.scope.current_kind == NodeKind.CLASS else NodeKind.FUNCTION
        self.scope.push(node.name, kind)

        old_locals = self.local_assignments.copy()
        self.local_assignments.clear()

        # Process assignments first for better resolution
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                self._track_assignment(child)
            elif isinstance(child, ast.AnnAssign):
                self._track_annotated_assignment(child)

        self.generic_visit(node)

        self.local_assignments = old_locals
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call."""
        caller_fqn = self.scope.current_fqn

        # Resolve the call target
        target, confidence, is_dynamic = self._resolve_call_target(node.func)

        if target:
            # Determine if call is conditional
            is_conditional = self._is_in_conditional(node)

            self.graph.add_edge(
                GraphEdge(
                    source_id=caller_fqn,
                    target_id=target,
                    kind=EdgeKind.CALLS,
                    location=SourceLocation(
                        file=self.file_path,
                        line_start=node.lineno,
                        line_end=_get_end_lineno(node, node.lineno),
                        col_start=node.col_offset,
                    ),
                    is_conditional=is_conditional,
                    is_dynamic=is_dynamic,
                    confidence=confidence,
                )
            )

        # Continue visiting for nested calls
        self.generic_visit(node)

    def _resolve_call_target(self, func: ast.AST) -> tuple[str | None, float, bool]:
        """
        Resolve call target to FQN.

        Returns:
            Tuple of (target_fqn, confidence, is_dynamic)
        """
        if isinstance(func, ast.Name):
            # Direct function call: foo()
            name = func.id

            # Check local symbol table first
            if name in self.symbol_table:
                return self.symbol_table[name], 1.0, False

            # Check if it's a builtin
            import builtins

            if hasattr(builtins, name):
                return f"builtins.{name}", 1.0, False

            # Unresolved - might be imported
            return name, 0.5, False

        elif isinstance(func, ast.Attribute):
            # Method call: obj.method()
            attr = func.attr
            value_name = self._get_value_name(func.value)

            if value_name:
                # Check if we know the type
                if value_name in self.local_assignments:
                    type_hint = self.local_assignments[value_name]
                    return f"{type_hint}.{attr}", 0.8, False

                # Check if it's a known module
                if value_name in self.symbol_table:
                    return f"{self.symbol_table[value_name]}.{attr}", 0.9, False

                # Method call on unknown object
                return f"{value_name}.{attr}", 0.4, False

            # Dynamic attribute access
            return f"<dynamic>.{attr}", 0.3, True

        elif isinstance(func, ast.Call) and self.include_dynamic_calls:
            # Call on return value: foo()()
            return None, 0.0, True

        elif isinstance(func, ast.Subscript) and self.include_dynamic_calls:
            # Call on subscript: foo[0]()
            return None, 0.0, True

        return None, 0.0, True

    def _get_value_name(self, node: ast.AST) -> str | None:
        """Get the base name from an expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_value_name(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_value_name(node.func)
        return None

    def _track_assignment(self, node: ast.Assign) -> None:
        """Track variable assignments for type inference."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Try to infer type from value
                type_hint = self._infer_type(node.value)
                if type_hint:
                    self.local_assignments[target.id] = type_hint

    def _track_annotated_assignment(self, node: ast.AnnAssign) -> None:
        """Track annotated assignments."""
        if isinstance(node.target, ast.Name) and node.annotation:
            type_hint = self._get_annotation_str(node.annotation)
            self.local_assignments[node.target.id] = type_hint

    def _infer_type(self, node: ast.AST) -> str | None:
        """Infer type from expression."""
        if isinstance(node, ast.Call):
            # Constructor call: Foo()
            func_name = self._get_value_name(node.func)
            if func_name and func_name[0].isupper():
                return func_name
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Set):
            return "set"
        elif isinstance(node, ast.Constant):
            return type(node.value).__name__
        return None

    def _get_annotation_str(self, node: ast.AST) -> str:
        """Convert annotation to string."""
        try:
            return ast.unparse(node)
        except Exception:
            return "<unknown>"

    def _is_in_conditional(self, node: ast.AST) -> bool:
        """
        Check if node is inside a conditional context.

        This is a simplified check - a full implementation would
        track the AST path.
        """
        # This would require parent tracking in the AST
        # For now, we'll mark all calls as non-conditional
        return False


class ASTExtractor(CallGraphExtractor):
    """
    Call graph extractor using Python's built-in AST module.

    This is the default extractor that is always available without
    external dependencies.
    """

    def extract_file(self, file_path: Path) -> CallGraph:
        """Extract call graph from a file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ParseError(f"Cannot decode {file_path}: {e}") from e

        module_name = file_path.stem
        return self.extract_module(source, module_name, str(file_path))

    def extract_module(
        self,
        source: str,
        module_name: str = "<module>",
        file_path: str = "<string>",
    ) -> CallGraph:
        """Extract call graph from source string."""
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            raise ParseError(f"Syntax error in {file_path}: {e}") from e

        # Pass 1: Collect definitions
        def_visitor = DefinitionVisitor(
            file_path=file_path,
            module_name=module_name,
            include_lambdas=self.include_lambdas,
            include_comprehensions=self.include_comprehensions,
        )
        def_visitor.visit(tree)

        # Pass 2: Collect calls
        call_visitor = CallVisitor(
            file_path=file_path,
            module_name=module_name,
            graph=def_visitor.graph,
            symbol_table=def_visitor.symbol_table,
            include_dynamic_calls=self.include_dynamic_calls,
        )
        call_visitor.visit(tree)

        return def_visitor.graph

    @property
    def backend_name(self) -> str:
        return "ast"

    @property
    def backend_version(self) -> str:
        return f"Python {sys.version_info.major}.{sys.version_info.minor}"
