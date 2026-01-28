"""
LPython ASR-based call graph extractor.

Uses LPython's Abstract Semantic Representation (ASR) for extraction.
ASR provides richer semantic information than Python's AST, including:
- Full type information (requires type annotations)
- Resolved imports
- Semantic rather than syntactic representation

This extractor requires LPython to be installed:
    conda install -c conda-forge lpython

Or built from source:
    https://github.com/lcompilers/lpython
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .extractor import CallGraphExtractor, ParseError, UnsupportedFeatureError
from .models import (
    CallGraph,
    EdgeKind,
    FunctionSignature,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
)


class LPythonNotFoundError(ImportError):
    """LPython is not installed or not in PATH."""
    pass


def _check_lpython() -> str:
    """
    Check if LPython is available.

    Returns:
        Path to lpython executable

    Raises:
        LPythonNotFoundError: If lpython is not found
    """
    import shutil

    lpython_path = shutil.which("lpython")
    if lpython_path is None:
        raise LPythonNotFoundError(
            "LPython is not installed or not in PATH. "
            "Install via: conda install -c conda-forge lpython"
        )
    return lpython_path


def _get_lpython_version() -> str:
    """Get LPython version string."""
    try:
        result = subprocess.run(
            ["lpython", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or result.stderr.strip() or "unknown"
    except Exception:
        return "unknown"


class ASRExtractor(CallGraphExtractor):
    """
    Call graph extractor using LPython's ASR.

    ASR (Abstract Semantic Representation) provides richer semantic
    information than Python's AST, making call resolution more accurate.

    Note: LPython requires type-annotated Python code. Code without
    type annotations may fail to parse or produce incomplete ASR.
    """

    def __init__(self, lpython_path: Optional[str] = None, **kwargs):
        """
        Initialize ASR extractor.

        Args:
            lpython_path: Path to lpython executable (auto-detected if None)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)

        if lpython_path:
            self._lpython = lpython_path
        else:
            self._lpython = _check_lpython()

        self._version: Optional[str] = None

    def extract_file(self, file_path: Path) -> CallGraph:
        """Extract call graph from a file using LPython ASR."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        asr_json = self._generate_asr(file_path)
        return self._parse_asr(asr_json, file_path.stem, str(file_path))

    def extract_module(
        self,
        source: str,
        module_name: str = "<module>",
        file_path: str = "<string>",
    ) -> CallGraph:
        """Extract call graph from source string using LPython ASR."""
        import tempfile

        # Write to temp file (LPython requires file input)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(source)
            temp_path = Path(f.name)

        try:
            asr_json = self._generate_asr(temp_path)
            return self._parse_asr(asr_json, module_name, file_path)
        finally:
            temp_path.unlink()

    def _generate_asr(self, file_path: Path) -> Dict[str, Any]:
        """
        Generate ASR JSON from Python source using LPython.

        Args:
            file_path: Path to Python source file

        Returns:
            Parsed ASR as dictionary

        Raises:
            ParseError: If LPython fails to parse
        """
        try:
            result = subprocess.run(
                [self._lpython, "--show-asr", "--json", str(file_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            raise ParseError(f"LPython timed out parsing {file_path}")
        except FileNotFoundError:
            raise LPythonNotFoundError(f"LPython not found at {self._lpython}")

        if result.returncode != 0:
            # Check for common errors
            stderr = result.stderr

            if "type annotation" in stderr.lower():
                raise UnsupportedFeatureError(
                    f"LPython requires type annotations: {stderr}"
                )

            raise ParseError(f"LPython failed to parse {file_path}: {stderr}")

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid ASR JSON from LPython: {e}")

    def _parse_asr(
        self,
        asr: Dict[str, Any],
        module_name: str,
        file_path: str,
    ) -> CallGraph:
        """
        Parse ASR JSON into CallGraph.

        ASR structure (simplified):
        {
            "asr": {
                "TranslationUnit": {
                    "global_scope": {...},
                    "items": [...]
                }
            }
        }
        """
        graph = CallGraph()

        # Add module node
        graph.add_node(GraphNode(
            id=module_name,
            kind=NodeKind.MODULE,
            name=module_name,
            location=SourceLocation(file=file_path, line_start=1, line_end=1),
        ))

        # Parse ASR structure
        asr_root = asr.get("asr", {})
        translation_unit = asr_root.get("TranslationUnit", {})

        # Extract from global scope
        global_scope = translation_unit.get("global_scope", {})
        self._parse_scope(graph, global_scope, module_name, file_path)

        return graph

    def _parse_scope(
        self,
        graph: CallGraph,
        scope: Dict[str, Any],
        parent_fqn: str,
        file_path: str,
    ) -> None:
        """
        Parse an ASR scope (module, class, or function).

        Extracts:
        - Function definitions
        - Class definitions
        - Variable declarations (for type tracking)
        """
        # Parse symbols in scope
        for name, symbol in scope.items():
            if not isinstance(symbol, dict):
                continue

            symbol_type = next(iter(symbol.keys()), None)

            if symbol_type == "Function":
                self._parse_function(graph, name, symbol["Function"], parent_fqn, file_path)

            elif symbol_type == "Class":
                self._parse_class(graph, name, symbol["Class"], parent_fqn, file_path)

            elif symbol_type == "Variable":
                # Track for type inference
                pass

    def _parse_function(
        self,
        graph: CallGraph,
        name: str,
        func_data: Dict[str, Any],
        parent_fqn: str,
        file_path: str,
    ) -> None:
        """Parse a function from ASR."""
        fqn = f"{parent_fqn}.{name}"

        # Extract location
        loc = func_data.get("loc", {})
        location = SourceLocation(
            file=file_path,
            line_start=loc.get("first_line", 1),
            line_end=loc.get("last_line", 1),
            col_start=loc.get("first_column", 0),
            col_end=loc.get("last_column", 0),
        )

        # Extract signature
        args = func_data.get("args", [])
        params = tuple(arg.get("name", "") for arg in args if isinstance(arg, dict))

        return_type = func_data.get("return_type")
        return_type_str = self._type_to_string(return_type) if return_type else None

        signature = FunctionSignature(
            name=name,
            params=params,
            return_type=return_type_str,
        )

        # Determine node kind
        kind = NodeKind.FUNCTION  # ASR doesn't distinguish methods in the same way

        graph.add_node(GraphNode(
            id=fqn,
            kind=kind,
            name=name,
            location=location,
            signature=signature,
        ))

        # Add defines edge
        graph.add_edge(GraphEdge(
            source_id=parent_fqn,
            target_id=fqn,
            kind=EdgeKind.DEFINES,
            location=location,
        ))

        # Parse function body for calls
        body = func_data.get("body", [])
        self._extract_calls(graph, body, fqn, file_path)

    def _parse_class(
        self,
        graph: CallGraph,
        name: str,
        class_data: Dict[str, Any],
        parent_fqn: str,
        file_path: str,
    ) -> None:
        """Parse a class from ASR."""
        fqn = f"{parent_fqn}.{name}"

        loc = class_data.get("loc", {})
        location = SourceLocation(
            file=file_path,
            line_start=loc.get("first_line", 1),
            line_end=loc.get("last_line", 1),
        )

        graph.add_node(GraphNode(
            id=fqn,
            kind=NodeKind.CLASS,
            name=name,
            location=location,
        ))

        graph.add_edge(GraphEdge(
            source_id=parent_fqn,
            target_id=fqn,
            kind=EdgeKind.DEFINES,
            location=location,
        ))

        # Parse class members
        members = class_data.get("members", {})
        self._parse_scope(graph, members, fqn, file_path)

    def _extract_calls(
        self,
        graph: CallGraph,
        body: List[Dict[str, Any]],
        caller_fqn: str,
        file_path: str,
    ) -> None:
        """Extract function calls from ASR statement list."""
        for stmt in body:
            if not isinstance(stmt, dict):
                continue

            stmt_type = next(iter(stmt.keys()), None)

            if stmt_type == "SubroutineCall":
                call_data = stmt["SubroutineCall"]
                self._add_call_edge(graph, call_data, caller_fqn, file_path)

            elif stmt_type == "FunctionCall":
                call_data = stmt["FunctionCall"]
                self._add_call_edge(graph, call_data, caller_fqn, file_path)

            elif stmt_type == "Assignment":
                # Check RHS for calls
                value = stmt["Assignment"].get("value", {})
                self._extract_calls_from_expr(graph, value, caller_fqn, file_path)

            elif stmt_type in ("If", "While", "For"):
                # Recurse into body
                inner_body = stmt[stmt_type].get("body", [])
                self._extract_calls(graph, inner_body, caller_fqn, file_path)

                # And else branch if present
                else_body = stmt[stmt_type].get("orelse", [])
                self._extract_calls(graph, else_body, caller_fqn, file_path)

    def _extract_calls_from_expr(
        self,
        graph: CallGraph,
        expr: Dict[str, Any],
        caller_fqn: str,
        file_path: str,
    ) -> None:
        """Extract calls from an ASR expression."""
        if not isinstance(expr, dict):
            return

        expr_type = next(iter(expr.keys()), None)

        if expr_type in ("FunctionCall", "SubroutineCall"):
            self._add_call_edge(graph, expr[expr_type], caller_fqn, file_path)

        # Recurse into sub-expressions
        for key, value in expr.get(expr_type, {}).items() if expr_type else []:
            if isinstance(value, dict):
                self._extract_calls_from_expr(graph, value, caller_fqn, file_path)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._extract_calls_from_expr(graph, item, caller_fqn, file_path)

    def _add_call_edge(
        self,
        graph: CallGraph,
        call_data: Dict[str, Any],
        caller_fqn: str,
        file_path: str,
    ) -> None:
        """Add a call edge from ASR call data."""
        # Extract target function name
        target_name = call_data.get("name")
        if not target_name:
            return

        loc = call_data.get("loc", {})
        location = SourceLocation(
            file=file_path,
            line_start=loc.get("first_line", 1),
            line_end=loc.get("last_line", 1),
        )

        graph.add_edge(GraphEdge(
            source_id=caller_fqn,
            target_id=target_name,
            kind=EdgeKind.CALLS,
            location=location,
            confidence=1.0,  # ASR has resolved the call
        ))

    def _type_to_string(self, type_data: Dict[str, Any]) -> str:
        """Convert ASR type to string representation."""
        if not isinstance(type_data, dict):
            return str(type_data)

        type_kind = next(iter(type_data.keys()), None)

        if type_kind == "Integer":
            return "int"
        elif type_kind == "Real":
            return "float"
        elif type_kind == "Complex":
            return "complex"
        elif type_kind == "Logical":
            return "bool"
        elif type_kind == "Character":
            return "str"
        elif type_kind == "List":
            inner = type_data["List"].get("type")
            inner_str = self._type_to_string(inner) if inner else "Any"
            return f"List[{inner_str}]"
        elif type_kind == "Dict":
            key = type_data["Dict"].get("key_type")
            val = type_data["Dict"].get("value_type")
            key_str = self._type_to_string(key) if key else "Any"
            val_str = self._type_to_string(val) if val else "Any"
            return f"Dict[{key_str}, {val_str}]"

        return str(type_kind)

    @property
    def backend_name(self) -> str:
        return "asr"

    @property
    def backend_version(self) -> str:
        if self._version is None:
            self._version = _get_lpython_version()
        return self._version
