# Releasing

## Version management

The version is the single source of truth in `pyproject.toml` and is mirrored
in `server.json` (MCP registry) and `manifest.json` (MCPB bundle). The
`make bump` target updates all three atomically.

## Release workflow

```bash
make release VERSION=0.3.0
```

This runs:

1. `sed` updates `pyproject.toml` version
2. Python script updates `server.json` and `manifest.json` (root + packages)
3. `uv lock` regenerates the lockfile
4. `git add` stages all four files
5. `git commit -m "release: v0.3.0"`
6. `git tag -a v0.3.0 -m "v0.3.0"`
7. `git push origin main --follow-tags`

The tag push triggers the CI pipeline:

```
tag push (v0.3.0)
    │
    ├── validate (tag ↔ pyproject.toml)
    │
    ├── CI gate (lint + test matrix 3.11/3.12/3.13 + integration)
    │
    ├── release.yml
    │   ├── build sdist + wheel
    │   ├── publish to PyPI (Trusted Publisher / OIDC)
    │   └── create GitHub Release with changelog
    │
    └── publish-mcp.yml
        ├── build + push Docker image to GHCR
        │   (tags: 0.3.0, 0.3, latest)
        ├── publish to MCP Registry (mcp-publisher)
        ├── publish to Smithery (if SMITHERY_ENABLED)
        └── build .mcpb bundle artifact
```

## Prerequisites (one-time)

### PyPI Trusted Publisher

Set up OIDC-based publishing so no API tokens are needed:

1. Go to [PyPI Trusted Publishers](https://pypi.org/manage/account/publishing/)
2. Add a new publisher for GitHub Actions
3. Create a GitHub environment named `pypi`

### MCP Registry

The `server.json` contains the registry namespace
`io.github.egoughnour/curate-ipsum`. The README includes the ownership
verification comment `<!-- mcp-name: io.github.egoughnour/curate-ipsum -->`.

### Smithery (optional)

Set repository variable `SMITHERY_ENABLED=true` and add secret
`SMITHERY_API_KEY` to enable Smithery publishing.

## Manual release

Each workflow can also be triggered manually via `workflow_dispatch`:

```bash
gh workflow run release.yml -f version=0.3.0
gh workflow run publish-mcp.yml -f version=0.3.0
```
