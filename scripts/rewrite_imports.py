#!/usr/bin/env python3
"""Bulk-rewrite flat module imports to curate_ipsum.xxx package imports.

Usage:
    python scripts/rewrite_imports.py --dry-run   # preview changes
    python scripts/rewrite_imports.py              # apply changes
"""

import re
import sys
from pathlib import Path

# All modules that moved into curate_ipsum/
FLAT_MODULES = sorted(
    [
        "adapters",
        "domains",
        "graph",
        "models",
        "parsers",
        "rag",
        "regions",
        "server",
        "storage",
        "synthesis",
        "theory",
        "tools",
        "verification",
    ],
    key=len,
    reverse=True,
)  # longest first to avoid partial matches

# Build alternation pattern: (verification|synthesis|storage|...)
MOD_ALT = "|".join(FLAT_MODULES)

# Patterns to match:
#   from <module> import ...
#   from <module>.<sub> import ...
#   import <module>
#   import <module>.<sub>
# But NOT:
#   from .<module> import ...  (relative imports)
#   from curate_ipsum.<module> import ...  (already converted)
IMPORT_FROM_RE = re.compile(
    rf"^(\s*from\s+)({MOD_ALT})((?:\.\w+)*\s+import\s)",
    re.MULTILINE,
)
IMPORT_BARE_RE = re.compile(
    rf"^(\s*import\s+)({MOD_ALT})(\b)",
    re.MULTILINE,
)


def rewrite_file(path: Path, dry_run: bool) -> list[str]:
    """Rewrite imports in a single file. Returns list of change descriptions."""
    text = path.read_text()
    changes: list[str] = []

    def replace_from(m: re.Match) -> str:
        prefix, mod, rest = m.group(1), m.group(2), m.group(3)
        old = m.group(0)
        new = f"{prefix}curate_ipsum.{mod}{rest}"
        if old != new:
            changes.append(f"  {old.strip()!r} → {new.strip()!r}")
        return new

    def replace_bare(m: re.Match) -> str:
        prefix, mod, rest = m.group(1), m.group(2), m.group(3)
        old = m.group(0)
        new = f"{prefix}curate_ipsum.{mod}{rest}"
        if old != new:
            changes.append(f"  {old.strip()!r} → {new.strip()!r}")
        return new

    new_text = IMPORT_FROM_RE.sub(replace_from, text)
    new_text = IMPORT_BARE_RE.sub(replace_bare, new_text)

    if changes and not dry_run:
        path.write_text(new_text)

    return changes


def main():
    dry_run = "--dry-run" in sys.argv
    root = Path(__file__).resolve().parent.parent

    # Process curate_ipsum/ and tests/
    targets = list((root / "curate_ipsum").rglob("*.py")) + list((root / "tests").rglob("*.py"))

    total = 0
    for path in sorted(targets):
        # Skip __pycache__
        if "__pycache__" in str(path):
            continue
        rel = path.relative_to(root)
        changes = rewrite_file(path, dry_run)
        if changes:
            label = "[DRY RUN] " if dry_run else ""
            print(f"{label}{rel} ({len(changes)} changes):")
            for c in changes:
                print(c)
            total += len(changes)

    mode = "would change" if dry_run else "changed"
    print(f"\nTotal: {total} imports {mode} across {len(targets)} files scanned.")


if __name__ == "__main__":
    main()
