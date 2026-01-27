# Architectural Vision: Graph-Theoretic Code Analysis Framework

## Executive Summary

Curate-Ipsum is not merely a mutation testing orchestrator - it's the foundation for a **hierarchical graph-spectral code analysis framework** that enables near-optimal reachability computation, supports multiple analysis backends (symbolic execution, SMT solving, concolic testing), and maintains robust, self-healing metadata infrastructure for codebases.

## Core Framework Components

### A. Graph Spectral Foundation

The system leverages spectral graph theory to decompose codebases into computationally optimal partitions:

#### A.1 Fiedler Vector Computation

**Input**: Dependency graph or call graph `G = (V, E)`

**Process**:
1. Construct the graph Laplacian `L = D - A` where:
   - `D` = degree matrix (diagonal)
   - `A` = adjacency matrix
2. Compute eigenvalues and eigenvectors of `L`
3. The **Fiedler vector** (eigenvector of second-smallest eigenvalue λ₂) provides optimal bipartition

**Output**: Natural partition boundary that minimizes edge cuts while balancing partition sizes

```python
# Conceptual implementation
import numpy as np
from scipy.sparse.linalg import eigsh

def compute_fiedler_vector(laplacian: np.ndarray) -> np.ndarray:
    """Compute the Fiedler vector for graph partitioning."""
    # Get two smallest eigenvalues/vectors (smallest is always 0)
    eigenvalues, eigenvectors = eigsh(laplacian, k=2, which='SM')
    fiedler_vector = eigenvectors[:, 1]  # Second eigenvector
    return fiedler_vector
```

#### A.2 Careful Partitioning Strategy

Partitioning must respect:
- **Semantic boundaries**: Module/package structure
- **Historical mutability**: Frequently-changed regions partition together
- **Computational efficiency**: Partitions sized for online recomputation

```
Partition Quality Criteria:
├── Minimize inter-partition edges (spectral cut)
├── Balance partition sizes (Fiedler sign distribution)
├── Respect semantic boundaries (module alignment)
└── Optimize for historical change patterns (temporal locality)
```

#### A.3 Graph Module Identification

After initial partitioning, identify **graph modules** (densely connected subgraphs):

```
Module Detection Pipeline:
1. Apply Fiedler partitioning recursively
2. Identify strongly connected components (SCCs)
3. Detect community structure within partitions
4. Augment with virtual sink/source nodes
```

#### A.4 Virtual Sink/Source Augmentation

For each module `M`:
- Add virtual **source node** `s_M` with edges to all entry points
- Add virtual **sink node** `t_M` with edges from all exit points

This enables:
- Uniform reachability queries
- Module-to-module flow analysis
- Boundary condition handling

```
    [s_M] ──→ [entry₁] ──→ ... ──→ [exit₁] ──→ [t_M]
      │         │                     │           ▲
      └──→ [entry₂] ──→ ... ──→ [exit₂] ─────────┘
```

#### A.5 Planar Subgraph Identification

**Key Insight**: Planar graphs admit O(n) reachability preprocessing with O(1) query time (Kameda's algorithm).

**Process**:
1. Identify maximal planar subgraph of each module
2. Extract **Kuratowski subgraphs** (K₅, K₃,₃) as non-planar core
3. Handle non-planar complement separately

```
Module Graph Decomposition:
├── Planar Subgraph (fast reachability)
│   └── O(n) preprocessing, O(1) query
└── Non-planar Complement
    ├── Kuratowski subgraphs (atomic units)
    └── Recursive decomposition
```

#### A.6 Hierarchical SCC Condensation

Alternating strategy:
1. **Condense**: Treat modules as SCCs, collapse to single nodes
2. **Partition**: Apply Fiedler to condensed graph
3. **Recurse**: Within each partition, repeat

```
Level 0: Full graph
    ↓ (SCC condensation)
Level 1: Module graph
    ↓ (Fiedler partition)
Level 2: Module clusters
    ↓ (SCC condensation within clusters)
Level 3: Sub-module graph
    ... (recursive until Kuratowski atoms)
```

### B. Robust Metadata Infrastructure

**Core Principle**: The metadata supporting the codebase must be more robust than the code itself - potentially self-healing.

#### B.1 Metadata Objectives

The infrastructure supports multiple simultaneous objectives:

| Objective | Metadata Required |
|-----------|-------------------|
| Reachability queries | Planar decomposition, SCC hierarchy |
| Compiler analysis | CFG, DFG, SSA form |
| Refactoring | Dependency graph, call graph, type hierarchy |
| Mutation testing | Reachability, coverage mapping |
| Symbolic execution | Path conditions, constraint sets |

#### B.2 Self-Healing Properties

```
Metadata Consistency Invariants:
├── Graph structure matches AST
├── Reachability indices valid after edit
├── Partition boundaries respect module structure
├── Historical stats reflect actual changes
└── Cross-references bidirectionally consistent
```

**Healing Mechanisms**:
- Incremental recomputation on file change
- Partition rebalancing when skew exceeds threshold
- Index invalidation and lazy rebuild
- Consistency checking on query

### C. Mutation Testing as Validation Layer

Mutation testing serves dual purposes:

1. **Test Quality Assessment**: Traditional role
2. **Metadata Validation**: Mutations that survive indicate potential metadata gaps

```
Mutation → Reachability Query → Expected Test Coverage
    ↓              ↓                    ↓
If survives:  Check if reachable   If reachable but
              from any test        not killed: metadata
                                   may be inconsistent
```

### D. Symbolic Execution Integration

#### D.1 KLEE/LLVM Pipeline

For code reducible to C/C++ LLVM-compatible form:

```
Source Code → LLVM IR → KLEE → Path Conditions → Z3/SMT
     ↓           ↓         ↓          ↓            ↓
  Curate    Bitcode    Symbolic   Constraint   SAT/UNSAT
  metadata  generation execution    sets       verdicts
```

**Container Architecture**:
```
┌─────────────────────────────────────────────────────┐
│                  Curate-Ipsum Core                  │
├─────────────────┬─────────────────┬─────────────────┤
│  KLEE Container │ Z3/SMT Container│ Mutation Tools  │
│  (concolic exec)│ (constraint     │ (mutmut, etc.)  │
│                 │  solving)       │                 │
└─────────────────┴─────────────────┴─────────────────┘
```

#### D.2 Beyond Z3: Robust Alternatives

| Solver | Strength | Use Case |
|--------|----------|----------|
| Z3 | General purpose | Default SMT backend |
| CVC5 | Theory combination | Complex type constraints |
| Boolector | Bitvector | Low-level code analysis |
| MathSAT | Interpolation | Abstraction refinement |
| Yices | Speed | Quick satisfiability checks |

### E. Mathematical Function Encoding

**Key Insight**: Some boolean logic intractable conditions become tractable as mathematical problems.

#### E.1 Path Condition Transformation

```python
# Boolean intractable:
if (x > 0 and x < 100 and x*x - 5*x + 6 == 0):
    # SAT solver struggles with nonlinear arithmetic

# Mathematical reformulation:
import sympy as sp
x = sp.Symbol('x')
condition = sp.And(x > 0, x < 100, sp.Eq(x**2 - 5*x + 6, 0))
solutions = sp.solve(x**2 - 5*x + 6, x)  # [2, 3]
# Filter by constraints: both in (0, 100) ✓
```

#### E.2 Differential/Root-Finding Approach

```python
from scipy.optimize import fsolve, root_scalar

def path_condition_as_function(x):
    """Convert path condition to root-finding problem."""
    # Original: x² - 5x + 6 = 0 AND x > 0 AND x < 100
    return x**2 - 5*x + 6

# Find roots
roots = []
for x0 in np.linspace(0.1, 99.9, 10):  # Initial guesses in valid range
    root, info, ier, msg = fsolve(path_condition_as_function, x0, full_output=True)
    if ier == 1 and 0 < root[0] < 100:
        roots.append(root[0])
```

#### E.3 SymPy Expression Encoding

```python
class PathCondition:
    """Encode path conditions as SymPy expressions."""

    def __init__(self):
        self.constraints: List[sp.Basic] = []
        self.symbols: Dict[str, sp.Symbol] = {}

    def add_constraint(self, expr: str):
        """Parse and add constraint."""
        parsed = sp.sympify(expr, locals=self.symbols)
        self.constraints.append(parsed)

    def solve(self) -> List[Dict[sp.Symbol, Any]]:
        """Attempt symbolic solution."""
        system = sp.And(*self.constraints)
        return sp.solve(system, list(self.symbols.values()), dict=True)

    def to_numeric(self) -> Callable:
        """Convert to numeric function for root-finding."""
        combined = sum((c - 0)**2 for c in self.constraints if c.is_Relational)
        return sp.lambdify(list(self.symbols.values()), combined)
```

### F. Graph Database Integration

#### F.1 Code Property Graph (CPG)

The CPG combines:
- Abstract Syntax Tree (AST)
- Control Flow Graph (CFG)
- Data Flow Graph (DFG)
- Call Graph (CG)

```
CPG Node Types:
├── AST nodes (syntax structure)
├── CFG nodes (control flow)
├── DFG edges (data dependencies)
├── CG edges (call relationships)
└── Custom: Fiedler partition membership
```

#### F.2 Backend Options

| Backend | Query Language | Strength |
|---------|---------------|----------|
| **Joern** | CPGQL | Purpose-built for CPG |
| **Neo4j** | Cypher | Mature, scalable |
| **JanusGraph** | Gremlin | Distributed |
| **TigerGraph** | GSQL | High performance |
| **Amazon Neptune** | Gremlin/SPARQL | Managed service |

#### F.3 Joern Integration

```python
# Joern query for reachability
REACHABILITY_QUERY = """
cpg.method.name("source_function")
   .repeat(_.caller)(_.until(_.method.name("sink_function")))
   .path
"""

# Query for Kuratowski subgraph detection
K33_DETECTION = """
cpg.method
   .filter(_.callOut.size >= 3)
   .filter(m => cpg.method.filter(_.callOut.contains(m)).size >= 3)
"""
```

#### F.4 Code Graph RAG MCP

**Architecture**:
```
┌─────────────────────────────────────────────────────┐
│                   MCP Interface                      │
├─────────────────────────────────────────────────────┤
│              Code Graph RAG Layer                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Embedding  │  │   Graph     │  │  Retrieval  │ │
│  │   Model     │  │   Index     │  │   Engine    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────┤
│           Graph Database (Neo4j/Joern)              │
└─────────────────────────────────────────────────────┘
```

**MCP Tools for Graph RAG**:
- `query_reachability(source, sink)` - O(1) lookup via precomputed index
- `find_partition(node)` - Return Fiedler partition membership
- `get_module_boundary(region)` - Return module entry/exit points
- `semantic_search(query)` - RAG over code graph

## Triviality Proxy Signals

Based on graph structure, triviality can be estimated:

| Signal | High Triviality | Low Triviality |
|--------|-----------------|----------------|
| **Fan-out** | High (many callees) | Low |
| **Fan-in** | Low (few callers) | High (many callers) |
| **Planarity** | Planar subgraph | Kuratowski core |
| **SCC membership** | Singleton SCC | Large SCC |
| **Fiedler value** | Near partition boundary | Core of partition |

```python
def compute_triviality(node: GraphNode, graph: CodeGraph) -> float:
    """Estimate triviality from graph structure."""
    fan_out = graph.out_degree(node)
    fan_in = graph.in_degree(node)
    in_planar = graph.is_in_planar_subgraph(node)
    scc_size = len(graph.get_scc(node))

    # High fan-out, low fan-in, planar, small SCC → trivial
    triviality = (
        0.3 * min(fan_out / 10, 1.0) +      # Fan-out contribution
        0.3 * (1 - min(fan_in / 10, 1.0)) +  # Fan-in contribution (inverted)
        0.2 * (1.0 if in_planar else 0.0) +  # Planarity bonus
        0.2 * (1 - min(scc_size / 50, 1.0))  # Small SCC bonus
    )
    return triviality
```

## Implementation Roadmap

### Phase 1: Graph Infrastructure
1. Call graph extraction (AST-based)
2. Laplacian construction
3. Fiedler vector computation
4. Basic partitioning

### Phase 2: Hierarchical Decomposition
5. SCC detection and condensation
6. Recursive partitioning
7. Planar subgraph identification
8. Virtual sink/source augmentation

### Phase 3: Reachability Index
9. Kameda preprocessing for planar subgraphs
10. Hierarchical index construction
11. O(1) query implementation
12. Incremental update support

### Phase 4: Analysis Backends
13. Joern/Neo4j integration
14. KLEE container setup
15. Z3 container setup
16. SymPy path condition encoding

### Phase 5: RAG and MCP
17. Code graph embedding
18. Semantic search index
19. MCP tool exposure
20. Self-healing metadata hooks

## Key References

- **Fiedler, M.** (1973). Algebraic connectivity of graphs
- **Kameda, T.** (1975). On the vector representation of the reachability in planar directed graphs
- **Kuratowski, K.** (1930). Sur le problème des courbes gauches en topologie
- **Joern Documentation**: Code Property Graphs for security analysis
- **KLEE**: Symbolic execution for C/C++
- **SMT-LIB**: Standard for SMT solvers
