"""Tests for ChromaVectorStore — uses embedded Chroma (no server needed)."""
import pytest
import tempfile
import os


def _have_chroma():
    try:
        import chromadb
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _have_chroma(), reason="chromadb not installed")


from quickverify.rag.vector_store import ChromaVectorStore


class TestChromaVectorStore:
    def setup_method(self):
        # Use /tmp to avoid mounted-filesystem permission issues on cleanup
        self._tmpdir = tempfile.mkdtemp(dir="/sessions/pensive-peaceful-planck/tmp")
        self.store = ChromaVectorStore(
            collection_name="test_collection",
            persist_dir=self._tmpdir,
        )

    def test_add_and_count(self):
        self.store.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadata=[{"kind": "function"}, {"kind": "class"}],
        )
        assert self.store.count() == 2

    def test_search(self):
        self.store.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            metadata=[{"kind": "function"}, {"kind": "class"}, {"kind": "function"}],
        )
        results = self.store.search(embedding=[1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # First result should be "a" (exact match)
        assert results[0].id == "a"
        assert results[0].score > 0.5

    def test_search_with_filter(self):
        self.store.add(
            ids=["a", "b", "c"],
            embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],
            metadata=[{"kind": "function"}, {"kind": "class"}, {"kind": "function"}],
        )
        results = self.store.search(
            embedding=[1.0, 0.0, 0.0],
            top_k=10,
            filters={"kind": "class"},
        )
        assert all(r.metadata.get("kind") == "class" for r in results)

    def test_delete(self):
        self.store.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadata=[{"kind": "fn"}, {"kind": "cls"}],
        )
        assert self.store.count() == 2
        self.store.delete(["a"])
        assert self.store.count() == 1

    def test_upsert_updates(self):
        self.store.add(
            ids=["a"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadata=[{"version": 1}],
        )
        self.store.add(
            ids=["a"],
            embeddings=[[0.0, 1.0, 0.0]],
            metadata=[{"version": 2}],
        )
        assert self.store.count() == 1
        results = self.store.search([0.0, 1.0, 0.0], top_k=1)
        assert results[0].metadata.get("version") == 2

    def test_empty_search(self):
        results = self.store.search([1.0, 0.0], top_k=5)
        assert results == []
