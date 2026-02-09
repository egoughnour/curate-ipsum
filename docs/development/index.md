# Development Guide

```{toctree}
:maxdepth: 2

testing
releasing
```

## Setup

```bash
git clone https://github.com/egoughnour/curate-ipsum.git
cd curate-ipsum
uv sync --all-extras
uv run pre-commit install
```

## Code quality

The project uses [ruff](https://docs.astral.sh/ruff/) for both linting and
formatting, configured for Python 3.10+ with a 120-character line length.

```bash
make lint          # ruff check
make fmt           # ruff format + fix
make check         # lint + typecheck
```

Pre-commit hooks run automatically on `git commit`:

- ruff format
- ruff lint (with auto-fix)
- uv lock check

## Project layout

```
curate-ipsum/
├── server.py              # MCP server entry point
├── tools.py               # Tool implementations
├── models.py              # Pydantic data models
├── adapters/              # Evidence adapter (mutation → beliefs)
├── domains/               # Domain-specific smoke tests
├── graph/                 # Call graph extraction + spectral analysis
├── parsers/               # Multi-framework mutation report parsers
├── rag/                   # RAG pipeline + vector store + embeddings
├── regions/               # Hierarchical code region model
├── storage/               # Persistent stores (SQLite, Kuzu, synthesis)
├── synthesis/             # CEGIS engine + genetic algorithm + LLM clients
├── theory/                # Belief revision + provenance + rollback
├── verification/          # Z3 + angr backends + CEGAR orchestrator
├── tests/                 # pytest test suite
├── docker/                # Dockerfiles + compose
└── docs/                  # Sphinx documentation (you are here)
```
