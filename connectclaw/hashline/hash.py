"""Hash computation — xxHash32 wrapper, per-line context-aware hash.

Ported from pi-hashline-edit/src/hashline/hash.ts (MIT).
"""

from __future__ import annotations

import re

import xxhash

from .config import get_hash_length

# ─── Hash Alphabet ──────────────────────────────────────────────────────────
#
# Custom 16-character alphabet. Deliberately excludes:
#   - Hex digits A-F (prevents confusion with hex literals in code)
#   - Visually confusable letters: D, G, I, L, O (look like 0, 6, 1, 1, 0)
#   - Common vowels A, E, I, O, U (prevents accidental English words)

NIBBLE_STR = "ZPMQVRWSNKTXJBYH"
HASH_ALPHABET_RE = re.compile(f"^[{NIBBLE_STR}]+$")

# Lines containing at least one alphanumeric character (not just punctuation).
RE_SIGNIFICANT = re.compile(r"[^\W_]")


# ─── xxHash32 ────────────────────────────────────────────────────────────────


def xxh32(input: str, seed: int = 0) -> int:
    """Return xxHash32 as an unsigned 32-bit integer."""
    h = xxhash.xxh32(seed=seed)
    h.update(input.encode("utf-8"))
    return h.intdigest()


# ─── Line Normalization ─────────────────────────────────────────────────────


def normalize_hash_input(line: str) -> str:
    """Normalize a line for hash input: strip \\r, trim trailing whitespace."""
    return line.replace("\r", "").rstrip()


# ─── Context-aware Hash ─────────────────────────────────────────────────────


def compute_hash_from_context(prev: str, curr: str, next: str) -> str:
    """Compute an N-char hash from a line and its immediate neighbors.

    Uses prev + "\\0" + curr + "\\0" + next as the hash input, so:
    - Distant edits don't invalidate anchors (only same/adjacent lines).
    - Adjacent-edit invalidation is intentional.
    - Silent collisions require the entire 3-line window to match.

    All three inputs must already be normalized via normalize_hash_input.
    Hash length is taken from config (default 2, configurable to 3-4).
    """
    length = get_hash_length()
    input_str = prev + "\0" + curr + "\0" + next
    h = xxh32(input_str)

    # Extract `length` nibbles from the low 4*length bits.
    result = ""
    for i in range(length - 1, -1, -1):
        result += NIBBLE_STR[(h >> (i * 4)) & 0x0F]
    return result


def compute_line_hash(file_lines: list[str], index: int) -> str:
    """Compute the N-char hash for a line at a given 0-based index.

    Neighbors outside the file boundaries use "" as their normalized value.
    """
    prev = normalize_hash_input(file_lines[index - 1] if index > 0 else "")
    curr = normalize_hash_input(file_lines[index])
    next = normalize_hash_input(
        file_lines[index + 1] if index < len(file_lines) - 1 else ""
    )
    return compute_hash_from_context(prev, curr, next)


# ─── Fuzzy Unicode Normalization ────────────────────────────────────────────

_FUZZY_SINGLE_QUOTES_RE = re.compile("[‘’‚‛]")
_FUZZY_DOUBLE_QUOTES_RE = re.compile("[“”„‟]")
_FUZZY_HYPHENS_RE = re.compile("[‐‑‒–—―−]")
_FUZZY_UNICODE_SPACES_RE = re.compile("[  -   　]")


def normalize_fuzzy_line(text: str) -> str:
    """Fuzzy-normalize text for anchor textHint validation.

    Converts Unicode smart quotes, hyphens, and spaces to ASCII equivalents
    so minor rendering differences don't cause false stale-anchor detection.
    """
    text = text.rstrip()
    text = _FUZZY_SINGLE_QUOTES_RE.sub("'", text)
    text = _FUZZY_DOUBLE_QUOTES_RE.sub('"', text)
    text = _FUZZY_HYPHENS_RE.sub("-", text)
    text = _FUZZY_UNICODE_SPACES_RE.sub(" ", text)
    return text


def is_fuzzy_equivalent_line(expected: str, actual: str) -> bool:
    """Check if two lines are equivalent after fuzzy Unicode normalization."""
    return normalize_fuzzy_line(expected) == normalize_fuzzy_line(actual)
