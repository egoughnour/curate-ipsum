# Quick Start

## Running the MCP server

After installing, start the server:

```bash
uv run curate-ipsum
```

The server communicates over stdin/stdout using the MCP protocol. Connect an MCP
client (Claude Desktop, Claude Code, etc.) and the tools become available.

## First workflow: mutation testing

1. **Detect your project's mutation framework:**

   Call `detect_frameworks` with your project's working directory. The server
   auto-detects Stryker, mutmut, cosmic-ray, poodle, or universalmutator.

2. **Run mutation tests:**

   Call `run_mutation_tests` with a command like `npx stryker run` or
   `mutmut run`. The server parses the report and stores the results.

3. **Check region metrics:**

   Call `get_region_metrics` to see mutation score and PID-like trending metrics
   for any code region (file, class, function, or line range).

## Second workflow: graph-spectral analysis

1. **Extract the call graph:**

   Call `extract_call_graph` with your Python project directory. The server
   builds a full call graph with SCC analysis and persists it to SQLite.

2. **Partition the codebase:**

   Call `compute_partitioning` to apply Fiedler spectral partitioning —
   recursively bipartitions the graph using the second eigenvector of the
   Laplacian.

3. **Query reachability:**

   Call `query_reachability` between two functions. Uses Kameda O(1) index on
   planar subgraphs with BFS fallback for non-planar edges.

## Third workflow: verified synthesis

1. **Add evidence and assertions to the theory:**

   Call `store_evidence` and `add_assertion` to build up a belief revision theory
   about your code.

2. **Synthesize a verified patch:**

   Call `synthesize_patch` — the CEGIS engine uses LLM candidates + genetic
   algorithm evolution + Z3 verification to produce a formally checked patch.

3. **Review provenance:**

   Call `get_provenance` and `why_believe` to inspect the evidence chain behind
   any assertion.

## Service stack (Docker Compose)

For the full experience with persistent ChromaDB and angr symbolic execution:

```bash
make docker-up-verify     # starts Chroma + angr runner
uv run curate-ipsum       # server connects to running services
```
