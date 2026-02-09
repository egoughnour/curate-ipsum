# Configuration

All configuration is via environment variables. See `.env.example` in the
repository root for a complete template.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CURATE_IPSUM_GRAPH_BACKEND` | `sqlite` | Graph store: `sqlite` or `kuzu` |
| `MUTATION_TOOL_DATA_DIR` | `.mutation_tool_data` | Persistent data directory |
| `MUTATION_TOOL_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `MUTATION_TOOL_STRYKER_REPORT` | `reports/mutation/mutation.json` | Default Stryker report path |
| `MUTATION_TOOL_PID_WINDOW` | `5` | PID metrics rolling window size |
| `MUTATION_TOOL_PID_DECAY` | `0.8` | PID metrics exponential decay factor |
| `CHROMA_HOST` | *(empty)* | ChromaDB server host. Empty = ephemeral in-process |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `LLM_BACKEND` | `mock` | Synthesis LLM backend: `mock`, `cloud`, `local` |

## Docker Compose services

The `docker/docker-compose.yml` defines two services:

**chroma** (always-on) — ChromaDB vector database for RAG. Exposes port 8000
with persistent volume.

**angr-runner** (profile: `verify`) — One-shot container for angr symbolic
execution. Memory-limited to 2 GB, CPU-limited to 2 cores.

```bash
# ChromaDB only
docker compose -f docker/docker-compose.yml up -d

# ChromaDB + angr runner
docker compose -f docker/docker-compose.yml --profile verify up -d

# Everything
docker compose -f docker/docker-compose.yml --profile all up -d
```

## Graph backends

### SQLite (default)

Zero-config. Creates `{data_dir}/{project}/graph.db` automatically.
Stores call graph nodes, edges, Kameda reachability index, and Fiedler
partitions.

### Kuzu

Requires the `graphdb` extra (`pip install "curate-ipsum[graphdb]"`).
Set `CURATE_IPSUM_GRAPH_BACKEND=kuzu`.

Kuzu provides a native graph database with Cypher queries. Useful for
larger codebases where relational queries become a bottleneck.
