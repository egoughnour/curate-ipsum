"""
Abstract base class for call graph extraction.

This module defines the interface that both Python AST and LPython ASR
extractors implement. The common interface allows swapping backends
while maintaining consistent behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import CallGraph


class ExtractorError(Exception):
    """Base exception for extraction errors."""

    pass


class ParseError(ExtractorError):
    """Error parsing source code."""

    pass


class UnsupportedFeatureError(ExtractorError):
    """Source contains features not supported by this extractor."""

    pass


class CallGraphExtractor(ABC):
    """
    Abstract base class for call graph extraction.

    Implementations:
    - ASTExtractor: Uses Python's built-in ast module (always available)
    - ASRExtractor: Uses LPython's ASR (requires LPython installation)

    The interface is designed to be semantic-representation-agnostic,
    allowing the same downstream code to work with either backend.
    """

    def __init__(
        self,
        include_lambdas: bool = True,
        include_comprehensions: bool = False,
        include_dynamic_calls: bool = True,
        resolve_imports: bool = True,
    ):
        """
        Initialize the extractor.

        Args:
            include_lambdas: Include lambda expressions as nodes
            include_comprehensions: Include list/dict/set comprehensions
            include_dynamic_calls: Try to resolve getattr/operator.attrgetter calls
            resolve_imports: Follow imports to build cross-module graph
        """
        self.include_lambdas = include_lambdas
        self.include_comprehensions = include_comprehensions
        self.include_dynamic_calls = include_dynamic_calls
        self.resolve_imports = resolve_imports

    @abstractmethod
    def extract_file(self, file_path: Path) -> CallGraph:
        """
        Extract call graph from a single file.

        Args:
            file_path: Path to Python source file

        Returns:
            CallGraph containing nodes and edges from the file

        Raises:
            ParseError: If the file cannot be parsed
            FileNotFoundError: If the file doesn't exist
        """
        pass

    @abstractmethod
    def extract_module(self, source: str, module_name: str = "<module>") -> CallGraph:
        """
        Extract call graph from source code string.

        Args:
            source: Python source code
            module_name: Name to use for the module node

        Returns:
            CallGraph containing nodes and edges

        Raises:
            ParseError: If the source cannot be parsed
        """
        pass

    def extract_files(self, file_paths: list[Path]) -> CallGraph:
        """
        Extract call graph from multiple files.

        Args:
            file_paths: List of paths to Python source files

        Returns:
            Combined CallGraph from all files
        """
        combined = CallGraph()

        for file_path in file_paths:
            try:
                file_graph = self.extract_file(file_path)
                self._merge_graphs(combined, file_graph)
            except (ParseError, FileNotFoundError) as e:
                # Log warning but continue with other files
                import logging

                logging.warning("Failed to extract %s: %s", file_path, e)

        return combined

    def extract_directory(
        self,
        directory: Path,
        pattern: str = "**/*.py",
        exclude: set[str] | None = None,
    ) -> CallGraph:
        """
        Extract call graph from all Python files in a directory.

        Args:
            directory: Root directory to search
            pattern: Glob pattern for file matching
            exclude: Set of directory/file names to exclude

        Returns:
            Combined CallGraph from all matching files
        """
        exclude = exclude or {"__pycache__", ".git", ".venv", "venv", "node_modules"}

        files: list[Path] = []
        for file_path in directory.glob(pattern):
            # Check if any parent directory is in exclude list
            if any(part in exclude for part in file_path.parts):
                continue
            if file_path.name in exclude:
                continue
            files.append(file_path)

        return self.extract_files(files)

    def _merge_graphs(self, target: CallGraph, source: CallGraph) -> None:
        """Merge source graph into target graph."""
        for node in source.nodes.values():
            target.add_node(node)
        for edge in source.edges:
            target.add_edge(edge)

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the extraction backend (e.g., 'ast', 'asr')."""
        pass

    @property
    @abstractmethod
    def backend_version(self) -> str:
        """Version of the extraction backend."""
        pass


def get_extractor(
    backend: str = "auto",
    **kwargs,
) -> CallGraphExtractor:
    """
    Factory function to get an appropriate extractor.

    Args:
        backend: Backend to use ('auto', 'ast', 'asr')
        **kwargs: Additional arguments passed to extractor constructor

    Returns:
        CallGraphExtractor instance

    Raises:
        ValueError: If requested backend is not available
    """
    if backend == "auto":
        # Try ASR first, fall back to AST
        try:
            from .asr_extractor import ASRExtractor

            return ASRExtractor(**kwargs)
        except ImportError:
            from .ast_extractor import ASTExtractor

            return ASTExtractor(**kwargs)

    elif backend == "ast":
        from .ast_extractor import ASTExtractor

        return ASTExtractor(**kwargs)

    elif backend == "asr":
        from .asr_extractor import ASRExtractor

        return ASRExtractor(**kwargs)

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'ast', or 'asr'.")
