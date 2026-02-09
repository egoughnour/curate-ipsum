# BRS CI/CD Pipeline

## Overview

The BRS repository uses GitHub Actions for continuous integration and release management. Tests gate all releases.

## Workflows

### `ci.yml` - Continuous Integration

**Triggers:** Push/PR to `main` or `master`

```
┌─────────────────────────────────────────────────────┐
│                      ci.yml                          │
├─────────────────────────────────────────────────────┤
│                                                      │
│  test (matrix: 3.9, 3.11, 3.12)  ───  REQUIRED      │
│    ├── pip install -e ".[dev]"                      │
│    ├── pytest tests/ -v --tb=short                  │
│    └── coverage report (3.11 only)                  │
│                                                      │
│  lint  ───────────────────────────  ADVISORY        │
│    ├── black --check --diff                         │
│    └── mypy (ignore-missing-imports)                │
│                                                      │
└─────────────────────────────────────────────────────┘
```

- **Tests are required** - CI fails if any test fails
- **Lint is advisory** - `continue-on-error: true`, shows warnings only

### `version.yml` - Manual Release

**Triggers:** Manual `workflow_dispatch` with version input (e.g., `v1.1.0`)

```
User triggers with version="v1.1.0"
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  1. Validate version format (vX.X.X)                │
│  2. Update version in 3 locations:                  │
│     - pyproject.toml: version = "1.1.0"             │
│     - brs/__init__.py: __version__ = "1.1.0"        │
│     - brs/__init__.py: Version: 1.1.0 (docstring)   │
│  3. Commit (if changes exist)                       │
│  4. Create/force tag                                │
│  5. Push to main + push tag                         │
└─────────────────────────────────────────────────────┘
         │
         ▼
    Triggers release.yml
```

**Idempotent handling:**
- `git diff --staged --quiet || git commit` - skips commit if no changes
- `git tag -fa` - force-creates tag even if exists
- `git push origin "$TAG" --force` - overwrites remote tag if exists

### `release.yml` - PyPI Release

**Triggers:** Tag push matching `v*.*.*`

```
Tag push (v1.1.0)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  test (matrix: 3.9, 3.11, 3.12)  ───  REQUIRED      │
│    └── Must pass before build                       │
└─────────────────────────────────────────────────────┘
         │ (needs: test)
         ▼
┌─────────────────────────────────────────────────────┐
│  build                                              │
│    ├── Extract version from tag                     │
│    ├── Update version in sources (idempotent)      │
│    ├── python -m build                              │
│    └── Upload artifact                              │
└─────────────────────────────────────────────────────┘
         │ (needs: build)
         ▼
┌─────────────────────────────────────────────────────┐
│  publish                                            │
│    └── pypa/gh-action-pypi-publish                  │
└─────────────────────────────────────────────────────┘
         │
         ▼
    Package on PyPI
```

## End-to-End Flow

### Path A: Via `version.yml`

```
1. User triggers workflow_dispatch(version="v1.1.0")
2. version.yml updates files, commits, tags, pushes
3. Tag push triggers release.yml
4. release.yml: test → build → publish
5. Package appears on PyPI
```

### Path B: Direct Tag Push

```
1. User: git tag -a v1.1.0 && git push origin v1.1.0
2. Tag push triggers release.yml
3. release.yml: test → build → publish
4. Package appears on PyPI
```

## Version Synchronization

Three locations are kept in sync:

| File | Pattern | Example |
|------|---------|---------|
| `pyproject.toml` | `^version = .*` | `version = "1.1.0"` |
| `brs/__init__.py` | `^__version__ = ".*"` | `__version__ = "1.1.0"` |
| `brs/__init__.py` | `^Version: .*` | `Version: 1.1.0` |

**sed commands used:**

```bash
VERSION="1.1.0"
sed -i "s/^version = .*/version = \"${VERSION}\"/" pyproject.toml
sed -i "s/^__version__ = \".*\"/__version__ = \"${VERSION}\"/" brs/__init__.py
sed -i "s/^Version: .*/Version: ${VERSION}/" brs/__init__.py
```

## Test Requirements

All releases are gated on:

- **Python 3.9** - tests pass
- **Python 3.11** - tests pass
- **Python 3.12** - tests pass

No release can happen if any test fails on any Python version.

## Lint Policy

Linting is advisory (non-blocking) to allow initial development velocity:

- `continue-on-error: true` on lint job
- Failures show as warnings in GitHub UI
- Can be tightened later by removing `continue-on-error`

## Files

| Workflow | Path |
|----------|------|
| CI | `.github/workflows/ci.yml` |
| Version | `.github/workflows/version.yml` |
| Release | `.github/workflows/release.yml` |

## Troubleshooting

### "Nothing to commit"
Re-running `version.yml` after version is already set. Safe - workflow continues.

### "Tag already exists"
Re-running after partial failure. Handled by `git tag -fa` and `--force` push.

### "Tests failed"
Release blocked. Fix tests and re-tag or re-run workflow.

### "Lint failed"
Advisory only. Release proceeds. Fix formatting with `black brs/ tests/`.
