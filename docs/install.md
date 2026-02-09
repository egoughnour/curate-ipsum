# Installation

## From PyPI

```bash
pip install curate-ipsum
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install curate-ipsum
```

### Optional extras

```bash
# Z3 SMT solver for formal verification (recommended)
pip install "curate-ipsum[verify]"

# RAG pipeline — ChromaDB + sentence-transformers
pip install "curate-ipsum[rag]"

# Graph-spectral analysis (scipy + networkx)
pip install "curate-ipsum[graph]"

# Cloud LLM synthesis (httpx)
pip install "curate-ipsum[synthesis]"

# Kuzu graph database backend
pip install "curate-ipsum[graphdb]"

# GPU-accelerated embeddings
pip install "curate-ipsum[embeddings-gpu]"

# Kitchen sink for development
pip install "curate-ipsum[dev,verify,rag,graph,synthesis,graphdb]"
```

## Docker

The Docker image includes the full server with the `all-MiniLM-L6-v2` embedding
model baked in — no Python installation required.

```bash
docker pull ghcr.io/egoughnour/curate-ipsum:latest
```

## From source (development)

```bash
git clone https://github.com/egoughnour/curate-ipsum.git
cd curate-ipsum
uv sync --extra dev --extra verify --extra rag --extra graph --extra synthesis --extra graphdb
uv run pre-commit install  # set up pre-commit hooks
```

## MCP client configuration

### Claude Desktop

Add to your `claude_desktop_config.json`:

::::{tab-set}
:::{tab-item} uvx (recommended)
```json
{
  "mcpServers": {
    "curate-ipsum": {
      "command": "uvx",
      "args": ["curate-ipsum"]
    }
  }
}
```
:::
:::{tab-item} Docker
```json
{
  "mcpServers": {
    "curate-ipsum": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "ghcr.io/egoughnour/curate-ipsum:latest"]
    }
  }
}
```
:::
:::{tab-item} pip install
```json
{
  "mcpServers": {
    "curate-ipsum": {
      "command": "curate-ipsum"
    }
  }
}
```
:::
::::

### Other MCP clients

Curate-Ipsum speaks the standard MCP stdio transport. Any client that supports
`command` + `args` configuration can launch it with:

```
command: curate-ipsum
```

Or equivalently: `uv run curate-ipsum`, `python -m server`, or `docker run -i --rm ghcr.io/egoughnour/curate-ipsum`.
