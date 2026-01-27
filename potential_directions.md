# Potential Directions for Curate-Ipsum

## 1. Multi-Tool Mutation Testing Integration

### 1.1 Active Python Mutation Tools

| Tool | Report Format | Integration Complexity | Value |
|------|---------------|----------------------|-------|
| **mutmut** | Custom HTML/text | Medium | High - most popular |
| **cosmic-ray** | Celery/SQLite backend | High | High - advanced features |
| **poodle** | JSON/HTML/text | Low | Medium - simple setup |
| **universalmutator** | Text-based | Medium | High - multi-language |

### 1.2 Proposed Report Parsers

```python
# New parser functions to add to tools.py:

def parse_mutmut_output(report_path: str) -> MutationStats:
    """Parse mutmut HTML or results database."""

def parse_cosmic_ray_output(session_path: str) -> MutationStats:
    """Parse cosmic-ray session database."""

def parse_poodle_output(report_path: str) -> MutationStats:
    """Parse poodle JSON report."""
```

### 1.3 Auto-Detection Strategy

```python
TOOL_SIGNATURES = {
    "stryker": ["mutation.json", "stryker-report.json"],
    "mutmut": [".mutmut-cache", "html/index.html"],
    "cosmic-ray": ["session.sqlite", ".cosmic-ray.toml"],
    "poodle": ["mutants/", "poodle.json"],
}
```

## 2. Enhanced Metrics System

### 2.1 Centrality Implementation

Currently stubbed at `0.5`. Potential approaches:

**Option A: Call Graph Centrality**
- Parse AST to build call graph
- Calculate PageRank or betweenness centrality
- Requires: `networkx`, static analysis

**Option B: Import-Based Centrality**
- Analyze module import relationships
- Score based on import frequency
- Lighter weight than full call graph

**Option C: Coverage-Weighted Centrality**
- Use coverage data to weight importance
- Regions covered by more tests = more central
- Requires: coverage.py integration

### 2.2 Triviality Detection

Currently stubbed at `0.5`. Potential approaches:

**Option A: Cyclomatic Complexity**
- Low complexity = high triviality
- Use `radon` or `mccabe` modules

**Option B: Mutant Difficulty**
- Easy-to-kill mutants = trivial code
- Hard-to-kill mutants = critical code
- Derived from historical mutation scores

**Option C: Test Coverage Density**
- High coverage + high kill rate = potentially trivial
- Low coverage + surviving mutants = complex

### 2.3 Extended PID Controller

Current implementation is basic. Enhancements:

```python
class EnhancedPIDComponents(BaseModel):
    p: float           # Current error
    i: float           # Accumulated error (with decay)
    d: float           # Rate of change
    setpoint: float    # Target mutation score (default 0.8?)
    output: float      # Recommended action intensity
    trend: str         # "improving" | "stable" | "degrading"
```

## 3. Test Framework Auto-Detection

### 3.1 Output Pattern Library

```python
TEST_FRAMEWORKS = {
    "pytest": {
        "patterns": [
            r"(\d+) passed",
            r"(\d+) failed",
            r"(\d+) error",
        ],
        "exit_codes": {0: "all_passed", 1: "tests_failed", 2: "interrupted"},
    },
    "unittest": {
        "patterns": [
            r"Ran (\d+) tests",
            r"OK",
            r"FAILED \(failures=(\d+)\)",
        ],
    },
    "nose": {...},
    "tox": {...},
}
```

### 3.2 Framework Detection

```python
def detect_framework(stdout: str, stderr: str) -> str:
    """Detect test framework from output patterns."""
    for framework, config in TEST_FRAMEWORKS.items():
        if any(p.search(stdout + stderr) for p in config["patterns"]):
            return framework
    return "generic"
```

## 4. Region Definition System

### 4.1 Region Types

```python
class RegionType(str, Enum):
    FILE = "file"           # Single file
    MODULE = "module"       # Python module/package
    CLASS = "class"         # Single class
    FUNCTION = "function"   # Single function
    LINES = "lines"         # Line range
    PATTERN = "pattern"     # Glob pattern
```

### 4.2 Region Specification

```python
class RegionSpec(BaseModel):
    id: str
    type: RegionType
    target: str  # File path, class name, etc.
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    include_nested: bool = True
```

## 5. Typed Patches for Larger Code Blocks

### 5.1 Patch Model

```python
class TypedPatch(BaseModel):
    """Represents a code modification suggestion."""
    patch_id: str
    region_id: str
    patch_type: Literal["test_addition", "mutation_fix", "coverage_gap"]
    priority: float  # 0.0 to 1.0
    original_code: str
    suggested_code: str
    rationale: str
    affected_mutants: List[str]  # Mutant IDs this patch addresses
```

### 5.2 Patch Generation Strategies

**Strategy A: Surviving Mutant Analysis**
- Identify surviving mutants
- Generate test stubs that would kill them
- Requires: AST analysis of mutations

**Strategy B: Coverage Gap Analysis**
- Find uncovered code regions
- Suggest tests based on function signatures
- Requires: coverage.py integration

**Strategy C: LLM-Assisted Patching**
- Feed surviving mutants to Claude/GPT
- Request test generation
- Requires: LLM API integration

### 5.3 Patch Application Workflow

```
1. Run mutation tests
2. Analyze surviving mutants
3. Generate TypedPatch suggestions
4. Present patches to developer
5. Developer selects/modifies patches
6. Apply patches
7. Re-run mutation tests
8. Track improvement in metrics
```

## 6. Reporting and Visualization

### 6.1 Report Formats

```python
class ReportFormat(str, Enum):
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    SARIF = "sarif"  # Static Analysis Results Interchange Format
```

### 6.2 Dashboard Metrics

- Mutation score trends over time
- Per-file mutation score heatmap
- PID component visualization
- Surviving mutant classification
- Test effectiveness scores

## 7. Integration Points

### 7.1 CI/CD Integration

```yaml
# GitHub Actions example
- name: Run Mutation Tests
  uses: curate-ipsum/action@v1
  with:
    tool: mutmut
    threshold: 0.8
    fail-on-regression: true
```

### 7.2 IDE Integration

- VSCode extension for mutation results
- Inline annotations for surviving mutants
- Quick-fix suggestions for test gaps

### 7.3 MCP Client Enhancements

- Streaming progress updates
- Cancellation support
- Partial results for long runs

## 8. Performance Optimizations

### 8.1 Incremental Mutation Testing

- Track file hashes between runs
- Only mutate changed files
- Reuse test results for unchanged code

### 8.2 Parallel Execution

```python
async def run_mutation_tests_parallel(
    projectId: str,
    files: List[str],
    max_workers: int = 4,
) -> MutationRunResult:
    """Run mutations in parallel across files."""
```

### 8.3 Caching Layer

```python
@lru_cache(maxsize=100)
def get_cached_region_metrics(
    project_id: str,
    commit_sha: str,
    region_id: str
) -> RegionMetrics:
    """Cache region metrics for repeated queries."""
```
