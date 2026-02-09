"""
Persistent storage layer for curate-ipsum.

Provides:
- SynthesisStore: JSONL persistence for synthesis results
- GraphStore: Abstract graph storage with SQLite and Kuzu backends
- IncrementalEngine: File-change detection for incremental graph updates
"""

from storage.graph_store import GraphStore, build_graph_store
from storage.synthesis_store import SynthesisStore

__all__ = [
    "SynthesisStore",
    "GraphStore",
    "build_graph_store",
]
