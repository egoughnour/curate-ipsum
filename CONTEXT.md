# Project Context

This document explains the directory structure, repository relationships, and naming conventions that may not be obvious without context.

## Directory Structure Explained

```
heal_thyself/                      ← User's workspace folder
├── reference_version/
│   └── curate-ipsum/              ← Original snapshot (read-only reference)
├── from_origin/
│   └── curate-ipsum/              ← Git origin clone (for diff comparison)
├── use_this/
│   └── curate-ipsum/              ← CANONICAL working copy (edit this one)
├── pr_files/                      ← Files to copy INTO brs repo
│   ├── brs_*.py                   ← Python source for BRS PR
│   ├── test_*.py                  ← Test files for BRS PR
│   └── *.yml                      ← GitHub workflow files for BRS PR
├── *.md                           ← Root-level docs (mirrors of use_this/)
├── DOCS_INDEX.md                  ← Documentation navigation guide
└── CONTEXT.md                     ← You are here
```

### Why Three curate-ipsum Directories?

| Directory | Purpose | Mutability |
|-----------|---------|------------|
| `reference_version/` | Original state snapshot for comparison | Read-only |
| `from_origin/` | Fresh git clone for diff against origin | Read-only |
| `use_this/` | **Canonical working copy** - all edits here | Read-write |

When in doubt, use `use_this/curate-ipsum/`.

### Why Root-Level MD Duplicates?

The root-level `.md` files are convenience copies of `use_this/curate-ipsum/*.md`. The canonical versions live in `use_this/curate-ipsum/`. Keep them in sync:

```bash
cp use_this/curate-ipsum/*.md .
```

## Repository Relationships

### BRS (Belief Revision System)

| Aspect | Value |
|--------|-------|
| GitHub repo | `github.com/egoughnour/brs` |
| PyPI package | `py-brs` (not `brs` - that was taken) |
| Import name | `import brs` |
| CLI command | `brs` |

**Why the name mismatch?** The PyPI name `brs` was already taken, so the package is published as `py-brs`, but the import and CLI remain `brs`.

### curate-ipsum

| Aspect | Value |
|--------|-------|
| GitHub repo | `github.com/egoughnour/curate-ipsum` |
| Purpose | Mutation testing orchestration MCP server |
| Depends on | `py-brs` (planned integration) |

## pr_files/ Explained

The `pr_files/` directory contains implementation files for a PR to the **BRS repository**. These are NOT part of curate-ipsum.

**To apply the PR:**

```bash
# Clone BRS repo (if not already)
git clone https://github.com/egoughnour/brs.git
cd brs

# Copy PR files
cp /path/to/heal_thyself/pr_files/brs_revision.py brs/revision.py
cp /path/to/heal_thyself/pr_files/brs_init.py brs/__init__.py
cp /path/to/heal_thyself/pr_files/brs_cli.py brs/cli.py
cp /path/to/heal_thyself/pr_files/test_contraction.py tests/test_contraction.py
cp /path/to/heal_thyself/pr_files/*.yml .github/workflows/

# Test
pytest tests/ -v
```

See `pr_files/PR_README.md` for full instructions.

## Session Paths

This project was developed in a Cowork session. Paths like `/sessions/blissful-vigilant-lamport/` are session-specific and won't exist elsewhere.

**Translation:**

| Session Path | Your System |
|--------------|-------------|
| `/sessions/.../mnt/heal_thyself/` | Your local `heal_thyself/` folder |
| `/sessions/.../brs_temp/` | Temporary; clone fresh from GitHub |

## Key Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                        GitHub                                │
├─────────────────────────────────────────────────────────────┤
│  egoughnour/brs          egoughnour/curate-ipsum            │
│       │                          │                          │
│       ▼                          ▼                          │
│  PyPI: py-brs            (not on PyPI yet)                  │
│       │                          │                          │
└───────┼──────────────────────────┼──────────────────────────┘
        │                          │
        ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     heal_thyself/                            │
├─────────────────────────────────────────────────────────────┤
│  pr_files/               use_this/curate-ipsum/             │
│  (BRS PR files)          (curate-ipsum working copy)        │
│       │                          │                          │
│       │                          ├── README.md              │
│       │                          ├── ROADMAP.md             │
│       │                          ├── brs_integration_plan.md│
│       │                          └── ...                    │
│       │                                                     │
│       └── brs_revision.py (copy to brs/revision.py)        │
│       └── brs_cli.py (copy to brs/cli.py)                  │
│       └── *.yml (copy to .github/workflows/)               │
└─────────────────────────────────────────────────────────────┘
```

## Common Tasks

### Start fresh with BRS

```bash
git clone https://github.com/egoughnour/brs.git
cd brs
pip install -e ".[dev]"
pytest tests/ -v
```

### Start fresh with curate-ipsum

```bash
git clone https://github.com/egoughnour/curate-ipsum.git
cd curate-ipsum
# See README.md for setup
```

### Apply pending BRS changes

```bash
cd /path/to/brs
cp /path/to/heal_thyself/pr_files/brs_*.py brs/
cp /path/to/heal_thyself/pr_files/test_*.py tests/
cp /path/to/heal_thyself/pr_files/*.yml .github/workflows/
pytest tests/ -v
git add -A && git commit -m "Add AGM contraction"
```

## Cold-Start Documentation

For zero-context session resumption, read these in order:

| Document | Purpose |
|----------|---------|
| `PROGRESS.md` | **Start here.** Current state, what's done, what's next. |
| `DECISIONS.md` | Every architectural decision with reasoning (D-001 through D-008). |
| `PHASE2_PLAN.md` | Active implementation plan: 9-step graph-spectral infrastructure. |
| `ROADMAP.md` | Full milestone tracker (M1–M7). |
| `architectural_vision.md` | Deep theory: Fiedler, Kameda, Kuratowski, spectral decomposition. |

## Version History

| Date | Change |
|------|--------|
| 2025-01-27 | Initial documentation structure |
| 2025-01-27 | Added AGM contraction to BRS (pr_files/) |
| 2025-01-27 | Added CI/CD pipeline with test gating |
| 2025-01-27 | Fixed cascade bug (nodes with alternate support survive) |
| 2026-02-08 | Added cold-start docs: PROGRESS.md, DECISIONS.md, PHASE2_PLAN.md |
