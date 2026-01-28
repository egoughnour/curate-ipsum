"""
Call graph extraction module for curate-ipsum.

This module provides tools for extracting call graphs from Python source code.
It supports two backends:

1. AST backend (always available):
   Uses Python's built-in ast module. Works with any Python code but
   has limited semantic information.

2. ASR backend (requires LPython):
   Uses LPython's Abstract Semantic Representation for richer semantic
   information. Requires type-annotated code.

Usage:
    from graph import get_extractor, CallGraph

    # Auto-select best available backend
    extractor = get_extractor()

    # Extract from file
    graph = extractor.extract_file(Path("module.py"))

    # Extract from directory
    graph = extractor.extract_directory(Path("src/"))

    # Query the graph
    for func in graph.functions():
        callees = graph.get_callees(func.id)
        print(f"{func.name} calls: {callees}")

    # Export to DOT
    print(graph.to_dot())
"""

from .models import (
    CallGraph,
    EdgeKind,
    FunctionSignature,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
)

from .extractor import (
    CallGraphExtractor,
    ExtractorError,
    ParseError,
    UnsupportedFeatureError,
    get_extractor,
)

from .ast_extractor import ASTExtractor

# ASR extractor is optional (requires LPython)
try:
    from .asr_extractor import ASRExtractor, LPythonNotFoundError
    HAS_LPYTHON = True
except ImportError:
    ASRExtractor = None  # type: ignore
    LPythonNotFoundError = ImportError  # type: ignore
    HAS_LPYTHON = False


__all__ = [
    # Models
    "CallGraph",
    "GraphNode",
    "GraphEdge",
    "NodeKind",
    "EdgeKind",
    "SourceLocation",
    "FunctionSignature",
    # Extractors
    "CallGraphExtractor",
    "ASTExtractor",
    "ASRExtractor",
    "get_extractor",
    # Exceptions
    "ExtractorError",
    "ParseError",
    "UnsupportedFeatureError",
    "LPythonNotFoundError",
    # Flags
    "HAS_LPYTHON",
]
