# Synergies: Mutation Testing Ecosystem Integration

## Tool-to-Tool Synergies

### 1. mutmut + Curate-Ipsum

**Compatibility**: High
**Integration Effort**: Low

mutmut is the most actively maintained Python mutation testing tool and aligns well with Curate-Ipsum's architecture:

| mutmut Feature | Curate-Ipsum Synergy |
|----------------|---------------------|
| JSONL result cache | Direct import into `runs.jsonl` |
| Incremental mode | Supports Curate-Ipsum's historical tracking |
| Pytest integration | Matches existing test command model |
| `--paths-to-mutate` | Maps to Curate-Ipsum regions |

**Integration Pattern**:
```python
# mutmut outputs to .mutmut-cache/
# Parse and convert to MutationRunResult
def parse_mutmut_cache(cache_path: Path) -> MutationRunResult:
    # mutmut stores results in SQLite-like format
    # Convert to Curate-Ipsum's model
```

**Synergy Value**: mutmut's incremental testing reduces CI time while Curate-Ipsum provides historical analysis.

### 2. cosmic-ray + Curate-Ipsum

**Compatibility**: Medium
**Integration Effort**: Medium-High

cosmic-ray offers advanced customization that complements Curate-Ipsum's metrics:

| cosmic-ray Feature | Curate-Ipsum Synergy |
|--------------------|---------------------|
| Custom operators | Enables domain-specific mutations |
| Distributed execution | Scales beyond single-machine runs |
| Session persistence | Aligns with run history model |
| Plugin architecture | Extensible mutation strategies |

**Integration Pattern**:
```python
# cosmic-ray uses sessions stored in SQLite
# Extract results from cr database
def parse_cosmic_ray_session(session_path: Path) -> MutationRunResult:
    import sqlite3
    conn = sqlite3.connect(session_path)
    # Query work_items and results tables
```

**Synergy Value**: cosmic-ray's distributed execution + Curate-Ipsum's PID tracking enables enterprise-scale mutation testing with quality metrics.

### 3. universalmutator + Curate-Ipsum

**Compatibility**: High
**Integration Effort**: Low

universalmutator's language-agnostic approach extends Curate-Ipsum beyond Python:

| universalmutator Feature | Curate-Ipsum Synergy |
|--------------------------|---------------------|
| Multi-language support | Polyglot project analysis |
| Regex-based mutations | Predictable output format |
| TCE filtering | Reduces noise in metrics |
| Custom rules | Project-specific mutations |

**Integration Pattern**:
```python
# universalmutator outputs mutant files
# Count by running tests against each
def run_universalmutator(
    source_file: str,
    output_dir: str,
    test_command: str
) -> MutationRunResult:
    # Generate mutants, test each, aggregate results
```

**Synergy Value**: Single Curate-Ipsum instance can track mutation metrics across Python, JavaScript, Go, etc.

### 4. poodle + Curate-Ipsum

**Compatibility**: High
**Integration Effort**: Very Low

poodle's simplicity makes it an ideal "light mode" option:

| poodle Feature | Curate-Ipsum Synergy |
|----------------|---------------------|
| JSON output | Direct parsing possible |
| TOML config | Matches Curate-Ipsum config style |
| Plugin system | Custom mutation types |
| Fast startup | Quick feedback loops |

**Integration Pattern**:
```python
# poodle can output JSON directly
def parse_poodle_json(report_path: Path) -> MutationRunResult:
    with open(report_path) as f:
        data = json.load(f)
    # Map poodle schema to MutationRunResult
```

**Synergy Value**: poodle for local dev, full tools for CI - unified metrics in Curate-Ipsum.

## Framework Synergies

### 5. pytest + coverage.py + Curate-Ipsum

**Synergy Type**: Test Ecosystem Integration

```
pytest --cov=myproject
    ↓
coverage.py generates .coverage
    ↓
Curate-Ipsum reads coverage to weight centrality/triviality
    ↓
Mutation tools use coverage to target uncovered code
    ↓
Metrics feedback improves test targeting
```

**Implementation**:
```python
def compute_coverage_weighted_centrality(
    coverage_data: CoverageData,
    call_graph: CallGraph
) -> Dict[str, float]:
    """Weight centrality by test coverage frequency."""
```

### 6. pre-commit + Curate-Ipsum

**Synergy Type**: Developer Workflow

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: mutation-score-check
        name: Check mutation score
        entry: curate-ipsum check --threshold 0.75
        language: python
        pass_filenames: false
```

**Synergy Value**: Prevent commits that degrade mutation score.

### 7. GitHub Actions + Curate-Ipsum

**Synergy Type**: CI/CD Integration

```yaml
jobs:
  mutation-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run mutation tests
        run: |
          curate-ipsum run-mutation-tests \
            --projectId=${{ github.repository }} \
            --commitSha=${{ github.sha }} \
            --tool=mutmut
      - name: Check regression
        run: |
          curate-ipsum check-regression \
            --projectId=${{ github.repository }} \
            --threshold=0.02  # Max 2% regression allowed
```

### 8. Codecov/Coveralls + Curate-Ipsum

**Synergy Type**: Metrics Aggregation

Both coverage and mutation score are quality indicators:

```
Coverage Score (lines executed) + Mutation Score (tests detect changes)
    ↓
Combined Quality Index = (coverage * 0.4) + (mutation * 0.6)
    ↓
Track over time in Curate-Ipsum
    ↓
Surface in Codecov/Coveralls alongside coverage
```

## Data Flow Synergies

### 9. Historical Analysis Pipeline

```
Git Commits → Curate-Ipsum Runs → Time Series DB → Grafana
     ↓              ↓                   ↓            ↓
  commitSha      RunResult          Prometheus    Dashboards
                  history            metrics      alerts
```

**Components**:
- Curate-Ipsum: Run execution + storage
- Prometheus: Metrics scraping from JSONL
- Grafana: Visualization + alerting

### 10. LLM-Assisted Test Generation

```
Surviving Mutants → Claude/GPT API → Test Suggestions → TypedPatch
        ↓               ↓                  ↓               ↓
   mutation.json    "Generate test    pytest code     Apply patch
                    to kill this      suggestions      re-run
                    mutant..."
```

**Implementation Sketch**:
```python
async def generate_test_for_surviving_mutant(
    mutant: SurvivingMutant,
    source_context: str,
) -> TypedPatch:
    prompt = f"""
    This mutant survived testing:
    Original: {mutant.original_code}
    Mutated:  {mutant.mutated_code}

    Generate a pytest test that would detect this mutation.
    """
    # Call LLM API, parse response into TypedPatch
```

## Ecosystem Integration Matrix

| Tool | Data In | Data Out | Bidirectional |
|------|---------|----------|---------------|
| mutmut | Report parsing | Run commands | No |
| cosmic-ray | Session DB | Run commands | No |
| poodle | JSON reports | Run commands | No |
| pytest | Test results | Test commands | No |
| coverage.py | Coverage data | - | No |
| GitHub Actions | Workflow triggers | Status checks | Yes |
| Prometheus | - | Metrics export | No |
| LLM APIs | Suggestions | Mutant data | Yes |

## Recommended Integration Priority

1. **Immediate**: mutmut parser (highest value, lowest effort)
2. **Short-term**: poodle parser (backup tool, simple)
3. **Medium-term**: coverage.py integration (enables centrality)
4. **Medium-term**: GitHub Actions integration (CI/CD)
5. **Long-term**: cosmic-ray parser (enterprise features)
6. **Long-term**: LLM test generation (experimental)
