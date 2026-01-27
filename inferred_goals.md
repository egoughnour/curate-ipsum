# Inferred Goals: Curate-Ipsum Project Analysis

## Evidence-Based Goal Inference

### Primary Indicators

1. **Project Name**: "Curate-Ipsum"
   - "Curate" suggests curation/organization of testing artifacts
   - "Ipsum" (Latin: "itself") implies self-referential testing - testing the tests

2. **MCP Server Architecture**
   - Designed for LLM tool integration
   - Suggests goal of AI-assisted test improvement

3. **PID Controller Metrics**
   - Control system approach to quality
   - Goal: automated quality feedback loops

4. **Region-Based Tracking**
   - Granular analysis at code region level
   - Goal: targeted test improvement

5. **Historical Run Storage**
   - JSONL persistence
   - Goal: trend analysis and regression detection

6. **Config References**
   - `model = "gpt-5.1-codex-max"`
   - `model_reasoning_effort = "xhigh"`
   - Goal: LLM-driven analysis with maximum reasoning

## Inferred Goal Hierarchy

### Tier 1: Core Mission

**Goal: Create an AI-orchestrated mutation testing platform that enables continuous test quality improvement through automated analysis and suggestion.**

Evidence:
- MCP server for LLM tool access
- PID metrics for control-loop quality tracking
- Historical storage for trend detection
- Region-based granularity for targeted action

### Tier 2: Operational Goals

#### 2.1 Unified Mutation Testing Interface

**Goal**: Abstract away mutation tool differences behind a single MCP interface.

Evidence:
- `tool: str = "stryker"` parameter suggests multi-tool support planned
- Generic command execution model works with any tool
- Report parsing separates from execution

**Current State**: Stryker only
**Target State**: mutmut, cosmic-ray, poodle, universalmutator

#### 2.2 Temporal Quality Tracking

**Goal**: Track mutation score evolution over time for regression detection.

Evidence:
- `RunHistory` model with timestamp-sorted runs
- `PID_WINDOW` and `PID_DECAY` configuration
- Historical queries by project/region

**Current State**: Basic history, manual analysis
**Target State**: Automated regression alerts, trend visualization

#### 2.3 Regional Code Analysis

**Goal**: Enable fine-grained analysis at sub-file granularity.

Evidence:
- `regionId` parameter on all operations
- `RegionMetrics` model with centrality/triviality
- Per-file breakdown in `MutationRunResult`

**Current State**: regionId tracking, stubbed metrics
**Target State**: Full centrality/triviality calculation, region definition DSL

### Tier 3: Tactical Goals

#### 3.1 Typed Patch Generation

**Goal**: Generate typed code patches that improve mutation scores.

Evidence:
- MCP architecture enables LLM-driven suggestions
- Per-file mutation stats enable targeted patches
- PID output could drive patch priority

**Inferred Implementation**:
```python
class TypedPatch(BaseModel):
    target_region: str
    patch_type: Literal["test", "code", "config"]
    priority: float  # Derived from PID output
    code_diff: str
    expected_impact: float  # Predicted score improvement
```

#### 3.2 Centrality-Weighted Testing

**Goal**: Prioritize mutations in central/critical code regions.

Evidence:
- `centrality: float` in RegionMetrics (stubbed at 0.5)
- Region-based run tracking
- PID control suggests optimization focus

**Inferred Implementation**:
- Call graph analysis for centrality
- Weight mutation importance by centrality
- Focus test generation on high-centrality low-score regions

#### 3.3 Triviality Detection

**Goal**: Identify trivial code that doesn't need extensive testing.

Evidence:
- `triviality: float` in RegionMetrics (stubbed at 0.5)
- Separate from centrality (different concern)
- Enables resource optimization

**Inferred Implementation**:
- Cyclomatic complexity analysis
- Mutation difficulty scoring
- De-prioritize trivial regions in reports

### Tier 4: Emergent Goals

#### 4.1 Self-Healing Test Suite

**Goal**: Automatically generate tests that kill surviving mutants.

Inference Chain:
1. Track surviving mutants per region
2. Feed to LLM via MCP
3. LLM generates test suggestions
4. Apply as TypedPatch
5. Re-run mutations
6. Track improvement via PID

#### 4.2 Quality Gate Automation

**Goal**: Enforce mutation score thresholds in CI/CD.

Inference Chain:
1. `extended_timeout` addition suggests CI environments
2. Historical tracking enables baseline comparison
3. PID `d_term` detects quality degradation
4. Fail builds on regression

#### 4.3 Multi-Repository Tracking

**Goal**: Track mutation metrics across an organization's repos.

Evidence:
- `projectId` as first-class parameter
- Flexible data directory configuration
- JSON-serializable results

## Gap Analysis

### Implemented vs Inferred

| Inferred Goal | Current State | Gap |
|---------------|---------------|-----|
| Multi-tool support | Stryker only | Need: parsers for mutmut, cosmic-ray |
| Centrality calculation | Stubbed (0.5) | Need: call graph analysis |
| Triviality detection | Stubbed (0.5) | Need: complexity analysis |
| Typed patches | Not implemented | Need: LLM integration layer |
| Quality gates | Not implemented | Need: threshold checking, CI hooks |
| Visualization | Not implemented | Need: HTML reports, dashboards |

### Critical Path to Goals

```
Current State
    ↓
[1] Add mutmut parser (expand tool support)
    ↓
[2] Implement centrality (call graph via AST)
    ↓
[3] Implement triviality (radon integration)
    ↓
[4] Add threshold checking (quality gates)
    ↓
[5] Create TypedPatch model (patch framework)
    ↓
[6] LLM integration for patch generation
    ↓
Target State: Self-improving test suite
```

## Hypothesis: The "Heal Thyself" Connection

The folder name `heal_thyself` combined with project analysis suggests a meta-goal:

**Hypothesis**: Curate-Ipsum is intended to be used on itself - a mutation testing tool that improves its own test suite.

Supporting Evidence:
- Located in `heal_thyself` directory
- Self-referential name ("ipsum" = "itself")
- MCP server architecture enables AI self-improvement
- PID controller suggests autonomous optimization

**Potential Workflow**:
1. Run mutations on Curate-Ipsum's own code
2. Track metrics in its own storage
3. Use its own MCP interface to get improvement suggestions
4. Apply suggestions to its own tests
5. Repeat (closed-loop improvement)

## Recommended Immediate Actions

Based on inferred goals:

1. **Validate Stryker Parser**: Ensure robust Stryker report handling
2. **Add mutmut Parser**: Most popular Python tool, high value
3. **Implement Basic Centrality**: Even naive file-level centrality adds value
4. **Add Regression Detection**: `d_term < 0` should trigger alerts
5. **Document TypedPatch Spec**: Define the patch model before implementing

## Open Questions for Confirmation

1. Is multi-language support (via universalmutator) a goal?
2. Should centrality use static analysis or runtime coverage?
3. What is the target deployment: CLI, CI service, or IDE plugin?
4. Is the LLM integration for analysis, generation, or both?
5. What mutation score threshold is considered "healthy"?
