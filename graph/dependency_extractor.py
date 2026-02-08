"""
Module-level dependency graph extraction.

Extracts import relationships between Python modules in a project
directory, producing a CallGraph with MODULE nodes and IMPORTS edges.
This complements the function-level call graph from ast_extractor.py
by providing a higher-level view of module dependencies.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    CallGraph,
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
)


class ImportInfo:
    """Information about a single import statement."""

    __slots__ = ("module", "names", "level", "line", "is_from_import")

    def __init__(
        self,
        module: Optional[str],
        names: Tuple[str, ...],
        level: int,
        line: int,
        is_from_import: bool,
    ):
        self.module = module
        self.names = names
        self.level = level  # 0 = absolute, 1+ = relative
        self.line = line
        self.is_from_import = is_from_import


def _module_name_from_path(file_path: Path, root: Path) -> str:
    """
    Derive a dotted module name from a file path relative to root.

    Examples:
        root/src/foo/bar.py → src.foo.bar
        root/src/foo/__init__.py → src.foo
    """
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return file_path.stem

    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else file_path.stem


def _resolve_relative_import(
    importing_module: str,
    import_module: Optional[str],
    level: int,
) -> Optional[str]:
    """
    Resolve a relative import to an absolute module name.

    Args:
        importing_module: FQN of the module containing the import
        import_module: The module name in the import statement (may be None for 'from . import X')
        level: Number of dots (1 = '.', 2 = '..', etc.)

    Returns:
        Resolved absolute module name, or None if unresolvable.
    """
    if level == 0:
        return import_module

    parts = importing_module.split(".")
    # Go up 'level' packages. level=1 means same package.
    # If importing_module is "a.b.c", level=1 → "a.b", level=2 → "a"
    if level > len(parts):
        return None  # Can't go above root

    base_parts = parts[: len(parts) - level]
    base = ".".join(base_parts)

    if import_module:
        return f"{base}.{import_module}" if base else import_module
    return base if base else None


# Standard library module names (Python 3.10+).
# We use sys.stdlib_module_names when available, else a fallback set.
_STDLIB_MODULES: Set[str] = getattr(sys, "stdlib_module_names", set()) or {
    "abc", "ast", "asyncio", "collections", "concurrent", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "decimal", "enum",
    "functools", "glob", "hashlib", "heapq", "html", "http",
    "importlib", "inspect", "io", "itertools", "json", "logging",
    "math", "multiprocessing", "operator", "os", "pathlib", "pickle",
    "pprint", "queue", "random", "re", "shutil", "signal", "socket",
    "sqlite3", "string", "struct", "subprocess", "sys", "tempfile",
    "textwrap", "threading", "time", "traceback", "typing", "unittest",
    "urllib", "uuid", "warnings", "xml", "zipfile",
}


def _is_stdlib_or_thirdparty(module_name: str, known_local: Set[str]) -> bool:
    """
    Determine if a module is stdlib/third-party (True) or local (False).

    A module is considered local if its top-level package matches a known
    local module name.
    """
    top_level = module_name.split(".")[0]
    if top_level in _STDLIB_MODULES:
        return True
    if top_level in known_local:
        return False
    # Heuristic: if not in known local modules, assume third-party.
    return True


def extract_imports_from_source(source: str, file_path: str = "<string>") -> List[ImportInfo]:
    """
    Extract all import statements from Python source code.

    Returns a list of ImportInfo objects, one per import statement.
    """
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    imports: List[ImportInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(ImportInfo(
                    module=alias.name,
                    names=(alias.asname or alias.name,),
                    level=0,
                    line=node.lineno,
                    is_from_import=False,
                ))
        elif isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            imports.append(ImportInfo(
                module=node.module,
                names=names,
                level=node.level or 0,
                line=node.lineno,
                is_from_import=True,
            ))

    return imports


class DependencyExtractor:
    """
    Extract module-level dependency graph from a Python project.

    Produces a CallGraph where:
    - Each module is a GraphNode with NodeKind.MODULE
    - Each import is a GraphEdge with EdgeKind.IMPORTS
    - Only local (intra-project) imports create edges
    - stdlib and third-party imports are excluded
    """

    def __init__(
        self,
        include_stdlib: bool = False,
        include_thirdparty: bool = False,
    ):
        """
        Args:
            include_stdlib: If True, include stdlib modules as nodes/edges.
            include_thirdparty: If True, include third-party modules as nodes/edges.
        """
        self.include_stdlib = include_stdlib
        self.include_thirdparty = include_thirdparty

    def extract_directory(
        self,
        directory: Path,
        pattern: str = "**/*.py",
        exclude: Optional[Set[str]] = None,
    ) -> CallGraph:
        """
        Build a module-level dependency graph from all Python files in a directory.

        Args:
            directory: Root directory of the Python project.
            pattern: Glob pattern for matching Python files.
            exclude: Directory/file names to skip.

        Returns:
            CallGraph with MODULE nodes and IMPORTS edges.
        """
        exclude = exclude or {"__pycache__", ".git", ".venv", "venv", "node_modules", ".tox", ".mypy_cache"}

        # Phase 1: Discover all local modules
        file_to_module: Dict[Path, str] = {}
        for file_path in sorted(directory.glob(pattern)):
            if any(part in exclude for part in file_path.parts):
                continue
            if file_path.name in exclude:
                continue
            module_name = _module_name_from_path(file_path, directory)
            file_to_module[file_path] = module_name

        known_local: Set[str] = set()
        for mod in file_to_module.values():
            known_local.add(mod.split(".")[0])
            known_local.add(mod)

        # Phase 2: Build graph
        graph = CallGraph()

        # Add module nodes
        for file_path, module_name in file_to_module.items():
            try:
                source = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            line_count = source.count("\n") + 1
            graph.add_node(GraphNode(
                id=module_name,
                kind=NodeKind.MODULE,
                name=module_name,
                location=SourceLocation(
                    file=str(file_path),
                    line_start=1,
                    line_end=line_count,
                ),
            ))

        # Phase 3: Extract imports and create edges
        for file_path, module_name in file_to_module.items():
            try:
                source = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            imports = extract_imports_from_source(source, str(file_path))

            for imp in imports:
                # Resolve the target module
                if imp.level > 0:
                    target = _resolve_relative_import(module_name, imp.module, imp.level)
                else:
                    target = imp.module

                if target is None:
                    continue

                # Determine if local, stdlib, or third-party
                is_external = _is_stdlib_or_thirdparty(target, known_local)

                if is_external and not (self.include_stdlib or self.include_thirdparty):
                    continue

                # Confidence based on import type
                if imp.is_from_import and "*" in imp.names:
                    confidence = 0.7  # Wildcard import
                elif imp.level > 0:
                    confidence = 0.95  # Relative import (likely accurate)
                else:
                    confidence = 1.0

                # Ensure target node exists (might be external or unresolved)
                if target not in graph.nodes:
                    # Try to find a package match (import foo.bar → foo.bar or foo)
                    found = False
                    parts = target.split(".")
                    for i in range(len(parts), 0, -1):
                        prefix = ".".join(parts[:i])
                        if prefix in graph.nodes:
                            target = prefix
                            found = True
                            break

                    if not found:
                        if is_external:
                            # Add external node if configured to include
                            graph.add_node(GraphNode(
                                id=target,
                                kind=NodeKind.MODULE,
                                name=target,
                                metadata={"external": True},
                            ))
                        else:
                            # Local module not found — might be a sub-import
                            # Try the top-level package
                            top = target.split(".")[0]
                            if top in graph.nodes:
                                target = top
                            else:
                                continue  # Skip unresolvable

                graph.add_edge(GraphEdge(
                    source_id=module_name,
                    target_id=target,
                    kind=EdgeKind.IMPORTS,
                    location=SourceLocation(
                        file=str(file_path),
                        line_start=imp.line,
                        line_end=imp.line,
                    ),
                    confidence=confidence,
                ))

        return graph

    def extract_file(self, file_path: Path) -> List[ImportInfo]:
        """
        Extract imports from a single Python file.

        Returns a list of ImportInfo objects.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return []

        return extract_imports_from_source(source, str(file_path))
