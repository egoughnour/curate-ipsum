# Testing

## Running tests

```bash
make test              # fast suite — no Docker, no model download
make test-all          # everything including integration tests
make test-integration  # only integration tests
make test-docker       # only Docker-dependent tests
make test-embedding    # only embedding model tests
make coverage          # fast suite + HTML coverage report
```

## Test markers

Tests are categorised with custom pytest markers:

`@pytest.mark.integration`
: Requires **both** Docker daemon and the embedding model. The most
  stringent tier.

`@pytest.mark.docker`
: Requires a running Docker daemon. Tests angr container execution and
  ChromaDB Docker mode.

`@pytest.mark.embedding`
: Requires the `all-MiniLM-L6-v2` model to be downloadable. Tests real
  embedding generation and semantic search.

Unmarked tests run without any external services — they use mock backends,
deterministic hash-based embeddings, and in-process Chroma.

## Test tiers

### Unit + end-to-end (636 tests, ~10s)

The fast suite covers every subsystem with real state flow. No mocks of Z3 or
the graph store — only the angr Docker backend and LLM clients are stubbed.

Key test files:

- `test_m5_end_to_end.py` — Z3 constraint solving, CEGAR orchestration,
  harness builder
- `test_m6_rag_end_to_end.py` — Chroma + embeddings + RAG pipeline + graph
  expansion
- `test_full_pipeline_end_to_end.py` — graph → SQLite → Chroma → RAG →
  CEGIS + Z3

### Integration (11 tests, ~60s)

Requires real infrastructure:

- Real `all-MiniLM-L6-v2` model downloaded from HuggingFace
- Real Docker daemon for angr container execution
- Real ChromaDB HTTP client mode

### Auto-skip

The `tests/conftest.py` plugin automatically skips infrastructure-gated tests
when the required services are unavailable:

```python
# Run only in full CI or local dev with Docker + model
pytest -m integration

# Skip all infrastructure tests (default in CI matrix)
pytest -m "not integration"
```

## Writing new tests

- Use the `chroma_store` fixture (unique collection name per test via UUID)
  to avoid collection leaking between tests
- Use `DeterministicEmbeddingProvider` (hash-based, 384-dim) for tests that
  don't need real embeddings
- Use `pytest.importorskip("module")` for optional dependency tests
