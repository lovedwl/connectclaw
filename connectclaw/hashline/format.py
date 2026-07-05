"""Format helpers — hashline region rendering, changed-line range computation.

Ported from pi-hashline-edit/src/hashline/format.ts (MIT).
"""

from __future__ import annotations

from .hash import compute_line_hash

# ─── Constants ──────────────────────────────────────────────────────────────

ANCHOR_CONTEXT_LINES = 2
ANCHOR_MAX_OUTPUT_LINES = 12


# ─── Affected-line computation ──────────────────────────────────────────────


def compute_affected_line_range(
    first_changed_line: int | None,
    last_changed_line: int | None,
    result_line_count: int,
    context_lines: int = ANCHOR_CONTEXT_LINES,
    max_output_lines: int = ANCHOR_MAX_OUTPUT_LINES,
) -> tuple[int, int] | None:
    """Compute the post-edit line range covering changed lines plus context.

    Returns None if the range (with context) exceeds the output budget,
    signalling that the LLM should re-read instead.
    """
    if first_changed_line is None or last_changed_line is None:
        return None

    start = max(1, first_changed_line - context_lines)
    end = min(result_line_count, last_changed_line + context_lines)

    if end < start:
        return None

    if end - start + 1 > max_output_lines:
        return None

    return (start, end)


# ─── Hashline Region Formatting ─────────────────────────────────────────────


def format_hashline_region(
    file_lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    """Format a range of lines as LINE#HASH:content.

    Line numbers are left-padded within the block for visual alignment.
    """
    line_number_width = len(str(end_line))
    out: list[str] = []
    for line_num in range(start_line, end_line + 1):
        line = file_lines[line_num - 1]
        h = compute_line_hash(file_lines, line_num - 1)
        padded = str(line_num).rjust(line_number_width)
        out.append(f"{padded}#{h}:{line}")
    return "\n".join(out)


# ─── Changed Line Range Computation ─────────────────────────────────────────


def compute_changed_line_range(
    original: str,
    result: str,
) -> tuple[int, int] | None:
    """Compute first/last changed line numbers between two document versions.

    Uses character-level diff to locate the changed span, then maps to line
    numbers in the result document for downstream anchor chaining.
    """
    if original == result:
        return None

    def _count_visible_lines(text: str) -> int:
        if not text:
            return 0
        lines = text.split("\n")
        return len(lines) - 1 if text.endswith("\n") else len(lines)

    if not original:
        return (1, _count_visible_lines(result))

    if result.startswith(original) and original.endswith("\n"):
        return (_count_visible_lines(original) + 1, _count_visible_lines(result))

    # Find first differing character
    first_diff = 0
    min_len = min(len(original), len(result))
    while first_diff < min_len and original[first_diff] == result[first_diff]:
        first_diff += 1
    if first_diff == min_len and len(original) == len(result):
        return None

    # Find last differing character
    last_orig = len(original) - 1
    last_res = len(result) - 1
    while last_orig >= first_diff and last_res >= first_diff and original[last_orig] == result[last_res]:
        last_orig -= 1
        last_res -= 1

    def _index_to_line(char_idx: int, text: str) -> int:
        line = 1
        for i in range(min(char_idx, len(text))):
            if text[i] == "\n":
                line += 1
        return line

    first_changed_line = _index_to_line(first_diff + 1, result)
    if last_res < first_diff:
        last_changed_line = 1 if not result else _count_visible_lines(result)
    elif first_diff == 0 and len(original) > 0 and result.endswith(original):
        last_changed_line = first_changed_line
    else:
        last_changed_line = _index_to_line(last_res + 1, result)

    return (first_changed_line, last_changed_line)
