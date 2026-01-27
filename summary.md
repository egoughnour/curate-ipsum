# Curate-Ipsum Project Summary

## Overview

**Curate-Ipsum** is an MCP (Model Context Protocol) server designed to orchestrate and track mutation testing workflows. It serves as a unified interface for running unit tests, integration tests, and mutation tests while maintaining historical metrics and enabling PID-based analysis of code quality over time.

## Architecture

The project consists of three core modules:

### 1. `models.py` - Data Models (79 lines)

Pydantic-based type-safe models defining the domain:

| Model | Purpose |
|-------|---------|
| `RunKind` | Enum for run types: `UNIT`, `INTEGRATION`, `MUTATION` |
| `RunMeta` | Base metadata: id, projectId, commitSha, regionId, timestamp |
| `TestRunResult` | Unit/integration test results with pass/fail counts |
| `FileMutationStats` | Per-file mutation statistics |
| `MutationRunResult` | Full mutation run results with per-file breakdown |
| `PIDComponents` | Proportional-Integral-Derivative components for metrics |
| `RegionMetrics` | Regional analysis with mutation score, centrality, triviality |
| `RunHistory` | Collection of runs for a project/region |

### 2. `server.py` - MCP Server Interface (167/155 lines)

FastMCP server exposing five tools:

| Tool | Type | Description |
|------|------|-------------|
| `run_unit_tests_tool` | async | Execute unit tests and capture results |
| `run_integration_tests_tool` | async | Execute integration tests and capture results |
| `run_mutation_tests_tool` | async | Execute mutation tests (Stryker) and parse reports |
| `get_run_history_tool` | sync | Retrieve historical runs for a project/region |
| `get_region_metrics_tool` | sync | Compute PID-like metrics for a region |

### 3. `tools.py` - Core Implementation (453/427 lines)

Business logic including:

- **Command Execution**: Async subprocess management with timeout support
- **Test Output Parsing**: Regex-based extraction of test results
- **Stryker Report Parsing**: JSON report processing with multi-format support
- **History Management**: JSONL-based persistent storage
- **PID Metrics Calculation**: Time-series analysis with configurable window and decay

## Version Differences

### Reference Version (Local)
- **Commit**: `d1510e6` - "added extended timeout as parameter for tools"
- **Extra Features**: `extended_timeout` parameter on all tool functions
- **Files Renamed**: `example_config.toml`, `example_mcp.json`
- **Data Present**: `.mutation_tool_data/runs.jsonl` with test run history

### Origin Version (Remote)
- **Commit**: `2c2353e` - "Add MIT License to the project"
- **Missing**: `extended_timeout` parameter functionality
- **Standard Files**: `config.toml`, `mcp.json`, `LICENSE`
- **No Data**: Clean checkout without run history

### Key Diff Summary (+46 lines in reference)
```
tools.py:  +34 lines (extended_timeout parsing and propagation)
server.py: +12 lines (extended_timeout in tool signatures)
```

## Configuration

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `MUTATION_TOOL_DATA_DIR` | `.mutation_tool_data` | Storage location |
| `MUTATION_TOOL_STRYKER_REPORT` | `reports/mutation/mutation.json` | Report path |
| `MUTATION_TOOL_PID_WINDOW` | `5` | PID calculation window |
| `MUTATION_TOOL_PID_DECAY` | `0.8` | PID decay factor |
| `MUTATION_TOOL_LOG_LEVEL` | `INFO` | Logging verbosity |

## Current Functionality Catalog

### Implemented
1. Unit test execution and result parsing
2. Integration test execution and result parsing
3. Mutation test execution with Stryker report parsing
4. Historical run storage (JSONL format)
5. Run history retrieval with filtering
6. PID-based region metrics calculation
7. Configurable timeouts (reference version only)

### Placeholder/Stub Values
- `centrality`: Hardcoded to `0.5`
- `triviality`: Hardcoded to `0.5`

### Supported Test Output Formats
```regex
Total tests:\s*(\d+).+Passed:\s*(\d+).+Failed:\s*(\d+)
Tests run:\s*(\d+)\s*,\s*Passed:\s*(\d+)\s*,\s*Failed:\s*(\d+)
```

### Supported Mutation Tools
- **Stryker** (primary, with report parsing)
- Generic tool passthrough (command execution only)

## Augmentation Opportunities

### High Priority
1. **Centrality/Triviality Calculation**: Currently stubbed at 0.5
2. **Additional Mutation Tool Parsers**: mutmut, cosmic-ray report formats
3. **Test Framework Detection**: Auto-detect pytest, unittest, nose output
4. **Report Export**: HTML/JSON summary reports

### Medium Priority
5. **Mutant Survival Analysis**: Track surviving mutants across runs
6. **Code Region Definition**: Formal region boundary specification
7. **Differential Analysis**: Compare mutation scores between commits
8. **Async History Operations**: Non-blocking file I/O

### Lower Priority
9. **Database Backend**: SQLite/PostgreSQL for run storage
10. **Webhook Notifications**: Post-run callbacks
11. **Rate Limiting**: Protect against runaway mutation runs
12. **Caching Layer**: Memoize region metrics calculations
