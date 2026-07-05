"""Diff generation and line-ending utilities.

Ported from pi-hashline-edit/src/edit-diff.ts (MIT).
"""

from __future__ import annotations

import difflib
import re

from .hash import compute_line_hash


# ─── Line ending detection / normalization ──────────────────────────────────


def detect_line_ending(content: str) -> str:
    """Detect the dominant line ending style. Returns '\\r\\n' or '\\n'."""
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1 or crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    """Normalize all line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    """Restore line endings from LF to the given style."""
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def has_mixed_line_endings(content: str) -> bool:
    """Return True when content mixes line-ending styles."""
    has_crlf = "\r\n" in content
    has_bare_lf = bool(re.search(r"(?<!\r)\n", content))
    has_lone_cr = bool(re.search(r"\r(?!\n)", content))
    style_count = sum([has_crlf, has_bare_lf, has_lone_cr])
    return style_count > 1


def strip_bom(content: str) -> tuple[str, str]:
    """Strip BOM if present. Returns (bom, text)."""
    if content.startswith("﻿"):
        return ("﻿", content[1:])
    return ("", content)


# ─── Diff generation ────────────────────────────────────────────────────────


def generate_diff_with_hashes(
    old_content: str,
    new_content: str,
) -> str:
    """Generate a diff with hashline anchors on new/modified lines.

    Uses difflib SequenceMatcher for line-level diffing.
    Returns a compact diff with LINE#HASH: anchors on context and added lines.
    """
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    # Remove trailing sentinel from split
    if old_content.endswith("\n"):
        old_lines = old_lines[:-1]
    if new_content.endswith("\n"):
        new_lines = new_lines[:-1]

    max_ln = max(len(old_lines), len(new_lines))
    ln_width = len(str(max_ln))

    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    output: list[str] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # Show context lines with hashes
            for offset in range(i2 - i1):
                new_idx = j1 + offset
                h = compute_line_hash(new_lines, new_idx)
                output.append(
                    f" {str(new_idx + 1).rjust(ln_width)}#{h}:{new_lines[new_idx]}"
                )
        elif tag == "replace":
            for idx in range(i1, i2):
                output.append(f"-{str(idx + 1).rjust(ln_width)}    {old_lines[idx]}")
            for idx in range(j1, j2):
                h = compute_line_hash(new_lines, idx)
                output.append(f"+{str(idx + 1).rjust(ln_width)}#{h}:{new_lines[idx]}")
        elif tag == "delete":
            for idx in range(i1, i2):
                output.append(f"-{str(idx + 1).rjust(ln_width)}    {old_lines[idx]}")
        elif tag == "insert":
            for idx in range(j1, j2):
                h = compute_line_hash(new_lines, idx)
                output.append(f"+{str(idx + 1).rjust(ln_width)}#{h}:{new_lines[idx]}")

    return "\n".join(output)


def generate_diff_string(
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> str:
    """Generate a standard unified diff (for details, without hashes)."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="a",
        tofile="b",
        n=context_lines,
    )
    return "".join(diff)
