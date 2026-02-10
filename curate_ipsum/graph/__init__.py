"""
Graph extraction and spectral analysis module for curate-ipsum.

This module provides:

1. **Call graph extraction** (Phase 1 — complete):
   - AST backend (always available)
   - ASR backend (requires LPython)
   - Dependency extraction (import-level)

2. **Graph-spectral analysis** (Phase 2):
   - Laplacian construction and Fiedler vector computation
   - Recursive spectral partitioning
   - Virtual sink/source augmentation
   - Hierarchical SCC condensation
   - Planarity testing and Kuratowski extraction
   - Kameda O(1) reachability index

Usage:
    from curate_ipsum.graph import get_extractor, CallGraph

    # Extract call graph
    extractor = get_extractor()
    graph = extractor.extract_directory(Path("src/"))

    # Spectral partitioning (requires scipy)
    from curate_ipsum.graph.spectral import compute_fiedler_components
    from curate_ipsum.graph.partitioner import GraphPartitioner

    partitioner = GraphPartitioner(min_partition_size=3)
    tree = partitioner.partition(graph)

    # O(1) reachability (requires networkx)
    from curate_ipsum.graph.planarity import check_planarity
    from curate_ipsum.graph.kameda import KamedaIndex

    result = check_planarity(graph)
    index = KamedaIndex.build(result.planar_subgraph, result.embedding)
    print(index.reaches("module.func_a", "module.func_b"))
"""

from .ast_extractor import ASTExtractor
from .dependency_extractor import DependencyExtractor
from .extractor import (
    CallGraphExtractor,
    ExtractorError,
    ParseError,
    UnsupportedFeatureError,
    get_extractor,
)
from .models import (
    CallGraph,
    EdgeKind,
    FunctionSignature,
    GraphEdge,
    GraphNode,
    NodeKind,
    SourceLocation,
)

# ASR extractor is optional (requires LPython)
try:
    from .asr_extractor import ASRExtractor, LPythonNotFoundError

    HAS_LPYTHON = True
except ImportError:
    ASRExtractor = None  # type: ignore
    LPythonNotFoundError = ImportError  # type: ignore
    HAS_LPYTHON = False

# Spectral analysis is optional (requires scipy)
try:
    from .spectral import (
        FiedlerResult,
        build_adjacency_matrix,
        build_laplacian,
        compute_fiedler,
        compute_fiedler_components,
        find_connected_components,
    )

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Planarity and Kameda are optional (requires networkx)
try:
    from .kameda import KamedaIndex
    from .planarity import (
        PlanarityResult,
        callgraph_to_networkx,
        check_planarity,
        networkx_to_callgraph,
    )

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Partitioner and hierarchy need scipy
try:
    from .hierarchy import HierarchyBuilder, HierarchyNode
    from .partitioner import GraphPartitioner, Partition, augment_partition
except ImportError:
    pass  # scipy not available


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
    "DependencyExtractor",
    "get_extractor",
    # Exceptions
    "ExtractorError",
    "ParseError",
    "UnsupportedFeatureError",
    "LPythonNotFoundError",
    # Spectral (optional: scipy)
    "FiedlerResult",
    "build_adjacency_matrix",
    "build_laplacian",
    "compute_fiedler",
    "compute_fiedler_components",
    "find_connected_components",
    # Partitioning (optional: scipy)
    "GraphPartitioner",
    "Partition",
    "augment_partition",
    # Hierarchy (optional: scipy)
    "HierarchyBuilder",
    "HierarchyNode",
    # Planarity (optional: networkx)
    "PlanarityResult",
    "check_planarity",
    "callgraph_to_networkx",
    "networkx_to_callgraph",
    # Kameda (optional: networkx)
    "KamedaIndex",
    # Flags
    "HAS_LPYTHON",
    "HAS_SCIPY",
    "HAS_NETWORKX",
]
