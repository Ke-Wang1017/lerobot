#!/usr/bin/env python
"""Lint check: flag destructive filesystem ops that lack a safety justification.

Rationale: a previous incident silently rmtree'd a user dataset because an
`except:` handler treated a load failure as "corruption" and recreated. To
prevent recurrence, every destructive call must be deliberately annotated.

A destructive call is OK when:
  - The same line, or the line above, contains: ``# safe-destruct: <reason>``
  - The reason explains *why* destruction is acceptable here (user-confirmed,
    temp-cleanup, our own metadata, etc.).

Run:
    python scripts/lint/no_silent_destruct.py path/to/file.py [...]

Exit code 0 if all destructive calls are annotated; non-zero otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Patterns that destroy data. Conservative — we'd rather false-positive
# (annotation required) than miss a footgun.
DESTRUCTIVE_PATTERNS = [
    r"\bshutil\.rmtree\s*\(",
    r"\.unlink\s*\(",  # Path.unlink, shm.unlink
    r"\bos\.remove\s*\(",
    r"\bos\.unlink\s*\(",
    r"\bos\.removedirs\s*\(",
    r"\.rmdir\s*\(",
    # shutil.move can overwrite an existing target — flag too
    r"\bshutil\.move\s*\(",
]

ANNOTATION_PATTERN = re.compile(r"#\s*safe-destruct:\s*\S+", re.IGNORECASE)


def check_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_number, line_text) for unannotated destructive ops."""
    violations: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []  # binary or unreadable — skip

    lines = text.splitlines()
    combined_pattern = re.compile("|".join(DESTRUCTIVE_PATTERNS))

    # Track docstring/triple-quoted-string state to skip mentions inside them.
    in_triple_double = False
    in_triple_single = False

    for i, line in enumerate(lines):
        # Update triple-quoted-string tracking (count occurrences on the line).
        # This is a best-effort heuristic — fine for our codebase.
        if not in_triple_single and line.count('"""') % 2 == 1:
            in_triple_double = not in_triple_double
        if not in_triple_double and line.count("'''") % 2 == 1:
            in_triple_single = not in_triple_single

        if in_triple_double or in_triple_single:
            continue

        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue

        if not combined_pattern.search(line):
            continue

        # Skip if the destructive pattern appears inside backticks (markdown-style doc reference).
        if re.search(r"`[^`]*(?:" + "|".join(DESTRUCTIVE_PATTERNS) + r")[^`]*`", line):
            continue

        # Check for annotation on this line
        if ANNOTATION_PATTERN.search(line):
            continue

        # Check the previous non-blank, non-comment line for the annotation
        annotated = False
        for j in range(i - 1, -1, -1):
            prev = lines[j].strip()
            if not prev:
                continue
            if prev.startswith("#"):
                if ANNOTATION_PATTERN.search(lines[j]):
                    annotated = True
                break
            # Hit code — annotation must be adjacent
            break

        if not annotated:
            violations.append((i + 1, line.rstrip()))

    return violations


def main(argv: list[str]) -> int:
    if not argv:
        # No files passed — scan src/ and scripts/ by default
        roots = [Path("src/lerobot"), Path("scripts")]
        files = [p for root in roots if root.is_dir() for p in root.rglob("*.py")]
    else:
        files = [Path(a) for a in argv]

    failed = False
    for path in files:
        if not path.is_file():
            continue
        violations = check_file(path)
        for line_no, line in violations:
            failed = True
            print(f"{path}:{line_no}: destructive op missing `# safe-destruct: <reason>`")
            print(f"    {line.strip()}")

    if failed:
        print()
        print("How to fix: add a comment on the same or previous line:")
        print("    # safe-destruct: <one-line reason this is safe>")
        print()
        print("Examples of acceptable reasons:")
        print("    # safe-destruct: user confirmed via GUI dialog")
        print("    # safe-destruct: our own temp dir, controlled path")
        print("    # safe-destruct: shm cleanup we created")
        print("    # safe-destruct: symlink update (not user data)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
