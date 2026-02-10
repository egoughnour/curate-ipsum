"""
RAG (Retrieval-Augmented Generation) subsystem for code-aware context (M6-deferred).

Provides vector store abstraction, embedding providers, and a search pipeline
that expands results using the project's existing GraphStore.

Decision: D-017
"""

from __future__ import annotations
