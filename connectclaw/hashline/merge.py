"""Three-way merge helper for stale-anchor recovery.

Ported from pi-hashline-edit/src/merge.ts (MIT).

Uses difflib.unified_diff + manual patch application with fuzzFactor 0.
Misaligned hunks are rejected, never slid — consistent with the strict
no-relocation principle.
"""

from __future__ import annotations

import difflib


def three_way_merge(
    base: str,
    base_edited: str,
    current: str,
) -> str | None:
    """Replay changes from base→base_edited onto current.

    Returns the merged text, or None when:
    - the patch cannot apply to current with fuzzFactor 0, or
    - the merged result is identical to current (nothing new to write).

    Short-circuit: if base == current, return base_edited directly.
    """
    if base == current:
        return base_edited

    # Generate patch from base → base_edited
    base_lines = base.splitlines(keepends=True)
    edited_lines = base_edited.splitlines(keepends=True)

    patch = difflib.unified_diff(
        base_lines,
        edited_lines,
        fromfile="a",
        tofile="b",
        n=3,
    )

    # Apply patch to current with fuzzFactor 0 (exact match only)
    merged = _apply_patch(current, patch, fuzz=0)
    if merged is None:
        return None

    if merged == current:
        return None

    return merged


def _apply_patch(
    original: str,
    patch_lines,
    fuzz: int = 0,
) -> str | None:
    """Apply a unified diff patch to original text with the given fuzz factor.

    Returns the patched text, or None if the patch cannot be applied.
    This is a simplified implementation that handles the common cases.
    """
    # Parse the unified diff into hunks
    hunks = _parse_hunks(patch_lines)

    orig_lines = original.split("\n")
    # Remove trailing empty element for non-newline-terminated files
    if original.endswith("\n") and orig_lines[-1] == "":
        orig_lines = orig_lines[:-1]

    result_lines = list(orig_lines)
    offset = 0  # tracks how inserted/deleted lines shift subsequent hunk positions

    for hunk in hunks:
        applied = _apply_hunk(result_lines, hunk, offset, fuzz)
        if applied is None:
            return None
        result_lines, hunk_offset = applied
        offset += hunk_offset

    return "\n".join(result_lines)


def _parse_hunks(patch_lines) -> list[dict]:
    """Parse unified diff into a list of hunk dicts."""
    hunks: list[dict] = []
    current: dict | None = None

    for line in patch_lines:
        line = line.rstrip("\n")
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            # Parse @@ -old_start,old_count +new_start,new_count @@
            match = __import__("re").match(
                r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line
            )
            if match:
                current = {
                    "old_start": int(match.group(1)),
                    "old_count": int(match.group(2)) if match.group(2) else 1,
                    "new_start": int(match.group(3)),
                    "new_count": int(match.group(4)) if match.group(4) else 1,
                    "lines": [],
                }
        elif current is not None:
            current["lines"].append(line)

    if current:
        hunks.append(current)

    return hunks


def _apply_hunk(
    result_lines: list[str],
    hunk: dict,
    offset: int,
    fuzz: int,
) -> tuple[list[str], int] | None:
    """Attempt to apply a single hunk. Returns (new_lines, offset_delta) or None."""
    old_start = hunk["old_start"] - 1  # 0-based
    # Adjust for previous hunks' offset
    adjusted_start = old_start + offset

    # Extract the expected context from the hunk
    hunk_lines: list[str] = hunk["lines"]
    expected_old: list[str] = []
    for hl in hunk_lines:
        if hl.startswith("-") or hl.startswith(" "):
            expected_old.append(hl[1:])

    # Try to find the match position with fuzz
    match_pos = _find_match(result_lines, adjusted_start, expected_old, fuzz)
    if match_pos is None:
        return None

    # Build replacement lines
    replacement: list[str] = []
    for hl in hunk_lines:
        if hl.startswith("+") or hl.startswith(" "):
            replacement.append(hl[1:])

    # Apply the replacement
    new_lines = (
        result_lines[:match_pos]
        + replacement
        + result_lines[match_pos + len(expected_old):]
    )

    offset_delta = len(replacement) - len(expected_old)
    return (new_lines, offset_delta)


def _find_match(
    lines: list[str],
    start: int,
    pattern: list[str],
    fuzz: int,
) -> int | None:
    """Find the position in lines where pattern matches, starting from `start`.

    With fuzz=0, requires exact match. Returns the match position or None.
    """
    # Clamp start to valid range
    start = max(0, min(start, len(lines)))

    # Try exact match at start
    if _lines_match(lines, start, pattern):
        return start

    if fuzz > 0:
        # Search within fuzz lines before and after
        for delta in range(1, fuzz + 1):
            if start - delta >= 0 and _lines_match(lines, start - delta, pattern):
                return start - delta
            if start + delta <= len(lines) and _lines_match(lines, start + delta, pattern):
                return start + delta

    return None


def _lines_match(lines: list[str], pos: int, pattern: list[str]) -> bool:
    """Check if pattern matches lines starting at pos."""
    if pos + len(pattern) > len(lines):
        return False
    for i, pl in enumerate(pattern):
        if lines[pos + i] != pl:
            return False
    return True
