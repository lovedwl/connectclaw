"""Parsing — anchor ref parsing, edit item validation, request normalization.

Ported from pi-hashline-edit/src/hashline/parse.ts + src/edit-normalize.ts (MIT).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Union

from .config import HASH_LENGTH_MAX, HASH_LENGTH_MIN, get_hash_length
from .hash import HASH_ALPHABET_RE, NIBBLE_STR

# ─── Display-prefix rejection regexes ──────────────────────────────────────
# These detect (and reject) hashline display prefixes inside edit payloads.
# They match ALL supported hash lengths, not just the session's.

_DISPLAY_HASH_QUANT = f"[{NIBBLE_STR}]{{{HASH_LENGTH_MIN},{HASH_LENGTH_MAX}}}"
_DISPLAY_PREFIX_RE = re.compile(
    rf"^\s*(?:>>>|>>)?\s*(?:\d+\s*#\s*|#\s*){_DISPLAY_HASH_QUANT}:"
)
_DISPLAY_PREFIX_PLUS_RE = re.compile(
    rf"^\+\s*(?:\d+\s*#\s*|#\s*){_DISPLAY_HASH_QUANT}:"
)
_DIFF_MINUS_RE = re.compile(r"^-\s*\d+\s{4}")


def _get_bare_prefix_re() -> re.Pattern:
    """Return a regex matching bare HH: prefixes at the current hash length."""
    return re.compile(rf"^\s*([{NIBBLE_STR}]{{{get_hash_length()}}}):")


# ─── Types ──────────────────────────────────────────────────────────────────

OpKind = Literal["replace", "append", "prepend", "replace_text"]


@dataclass
class Anchor:
    line: int
    hash: str
    text_hint: str | None = None


@dataclass
class ReplaceEdit:
    op: Literal["replace"]
    pos: Anchor
    lines: list[str]
    end: Anchor | None = None


@dataclass
class AppendEdit:
    op: Literal["append"]
    lines: list[str]
    pos: Anchor | None = None


@dataclass
class PrependEdit:
    op: Literal["prepend"]
    lines: list[str]
    pos: Anchor | None = None


@dataclass
class ReplaceTextEdit:
    op: Literal["replace_text"]
    oldText: str
    newText: str


HashlineEdit = Union[ReplaceEdit, AppendEdit, PrependEdit, ReplaceTextEdit]


# ─── Anchor Parsing ─────────────────────────────────────────────────────────


def _example_anchor() -> str:
    ln = get_hash_length()
    return f"5#{'MQQV'[:ln]}"


def _diagnose_line_ref(ref: str) -> str:
    trimmed = ref.strip()
    core = re.sub(r"^\s*[>+\-]*\s*", "", trimmed)
    example = _example_anchor()
    config_len = get_hash_length()

    if not core:
        return f'[E_BAD_REF] Invalid line reference "{ref}". Expected "LINE#HASH" (e.g. "{example}").'
    if re.match(r"^\d+\s*$", core):
        return f'[E_BAD_REF] Invalid line reference "{ref}": missing hash, use "LINE#HASH" from read output (e.g. "{example}").'
    if re.match(r"^\d+\s*:", core):
        return f'[E_BAD_REF] Invalid line reference "{ref}": wrong separator, use "LINE#HASH" instead of "LINE:...".'

    hash_match = re.match(r"^(\d+)\s*#\s*([^\s:]+)(?:\s*:.*)?$", core)
    if hash_match:
        line = int(hash_match.group(1))
        hash_str = hash_match.group(2)
        if line < 1:
            return f'[E_BAD_REF] Line number must be >= 1, got {line} in "{ref}".'
        if len(hash_str) != config_len:
            if (
                HASH_ALPHABET_RE.match(hash_str)
                and HASH_LENGTH_MIN <= len(hash_str) <= HASH_LENGTH_MAX
            ):
                return f'[E_BAD_REF] Invalid line reference "{ref}": hash length is {config_len} in this session, but this anchor has {len(hash_str)} characters — it looks like an anchor from a stale context or a different configuration. Re-read the file to get current anchors.'
            return f'[E_BAD_REF] Invalid line reference "{ref}": hash must be exactly {config_len} characters from {NIBBLE_STR} (e.g. "{example}").'
        if not HASH_ALPHABET_RE.match(hash_str):
            return f'[E_BAD_REF] Invalid line reference "{ref}": hash uses invalid characters, hashes use alphabet {NIBBLE_STR} only.'

    missing_hash_match = re.match(r"^(\d+)\s*#\s*$", core)
    if missing_hash_match:
        return f'[E_BAD_REF] Invalid line reference "{ref}": missing hash after "#", use "LINE#HASH" from read output.'

    if re.match(r"^0+\s*#", core):
        return f'[E_BAD_REF] Line number must be >= 1, got 0 in "{ref}".'

    return f'[E_BAD_REF] Invalid line reference "{trimmed or ref}". Expected "LINE#HASH" (e.g. "{example}").'


def parse_anchor_ref(ref: str) -> Anchor:
    """Parse a LINE#HASH[:content] reference into an Anchor.

    Tolerates leading ">+-" and whitespace (from mismatch/diff display)
    and an optional trailing ":content" display suffix preserved as text_hint.
    """
    core = re.sub(r"^\s*[>+\-]*\s*", "", ref).rstrip()
    match = re.match(r"^([0-9]+)\s*#\s*([^\s:]+)(?:\s*:(.*))?$", core, re.DOTALL)
    if not match:
        raise ValueError(_diagnose_line_ref(ref))

    line = int(match.group(1))
    if line < 1:
        raise ValueError(f'[E_BAD_REF] Line number must be >= 1, got {line} in "{ref}".')

    hash_str = match.group(2)
    config_len = get_hash_length()
    if len(hash_str) != config_len:
        if (
            HASH_ALPHABET_RE.match(hash_str)
            and HASH_LENGTH_MIN <= len(hash_str) <= HASH_LENGTH_MAX
        ):
            raise ValueError(
                f'[E_BAD_REF] Invalid line reference "{ref}": hash length is {config_len} '
                f"in this session, but this anchor has {len(hash_str)} characters — "
                f"it looks like an anchor from a stale context or a different configuration. "
                f"Re-read the file to get current anchors."
            )
        raise ValueError(
            f'[E_BAD_REF] Invalid line reference "{ref}": hash must be exactly '
            f'{config_len} characters from {NIBBLE_STR} (e.g. "{_example_anchor()}").'
        )

    if not HASH_ALPHABET_RE.match(hash_str):
        raise ValueError(
            f'[E_BAD_REF] Invalid line reference "{ref}": hash uses invalid '
            f"characters, hashes use alphabet {NIBBLE_STR} only."
        )

    text_hint = match.group(3)
    return Anchor(line=line, hash=hash_str, text_hint=text_hint if text_hint is not None else None)


# ─── Content Preprocessing ──────────────────────────────────────────────────


def _assert_no_display_prefixes(lines: list[str]) -> None:
    """Reject hashline display prefixes inside edit line payloads."""
    for line in lines:
        if not line:
            continue
        if _DISPLAY_PREFIX_RE.search(line) or _DISPLAY_PREFIX_PLUS_RE.search(line) or _DIFF_MINUS_RE.search(line):
            raise ValueError(
                f'[E_INVALID_PATCH] "lines" must contain literal file content, '
                f'not rendered "LINE#HASH:" or diff "+/-" prefixes. '
                f"Offending line: {line!r}"
            )


def _hashline_parse_text(edit: list[str] | None) -> list[str]:
    lines = edit or []
    _assert_no_display_prefixes(lines)
    return lines


# ─── Edit Item Validation ───────────────────────────────────────────────────

_ITEM_KEYS = {"op", "pos", "end", "lines", "oldText", "newText"}


def _assert_edit_item(edit: dict, index: int) -> None:
    unknown_keys = set(edit.keys()) - _ITEM_KEYS
    if unknown_keys:
        raise ValueError(
            f"Edit {index} contains unknown or unsupported fields: {', '.join(sorted(unknown_keys))}."
        )

    op = edit.get("op")
    if not isinstance(op, str):
        raise ValueError(f'Edit {index} requires an "op" string.')
    if op not in ("replace", "append", "prepend", "replace_text"):
        raise ValueError(
            f'[E_BAD_OP] Edit {index} uses unknown op "{op}". '
            f'Expected "replace", "append", "prepend", or "replace_text".'
        )

    if "pos" in edit and not isinstance(edit["pos"], str):
        raise ValueError(f'Edit {index} field "pos" must be a string when provided.')
    if "end" in edit and not isinstance(edit.get("end", ""), str):
        raise ValueError(f'Edit {index} field "end" must be a string when provided.')
    if "oldText" in edit and not isinstance(edit["oldText"], str):
        raise ValueError(f'Edit {index} field "oldText" must be a string when provided.')
    if "newText" in edit and not isinstance(edit.get("newText", ""), str):
        raise ValueError(f'Edit {index} field "newText" must be a string when provided.')
    if "lines" in edit and not (
        isinstance(edit["lines"], list)
        and all(isinstance(item, str) for item in edit["lines"])
    ):
        raise ValueError(f'Edit {index} field "lines" must be a string array.')

    if op == "replace_text":
        if not isinstance(edit.get("oldText"), str) or not isinstance(edit.get("newText"), str):
            raise ValueError(
                f'[E_BAD_OP] Edit {index} with op "replace_text" requires '
                f'string "oldText" and "newText" fields.'
            )
        if "pos" in edit or "end" in edit or "lines" in edit:
            raise ValueError(
                f'Edit {index} with op "replace_text" only supports "oldText" and "newText".'
            )
        return

    if "lines" not in edit:
        raise ValueError(f'Edit {index} requires a "lines" field.')

    if "oldText" in edit or "newText" in edit:
        raise ValueError(
            f'Edit {index} with op "{op}" does not support "oldText" or "newText".'
        )

    if op == "replace" and not isinstance(edit.get("pos"), str):
        raise ValueError(
            f'[E_BAD_OP] Edit {index} with op "replace" requires a "pos" anchor string.'
        )

    if op in ("append", "prepend") and "end" in edit:
        raise ValueError(
            f'[E_BAD_OP] Edit {index} with op "{op}" does not support "end". '
            f'Use "pos" or omit it for file boundary insertion.'
        )


def resolve_edit_anchors(edits: list[dict]) -> list[HashlineEdit]:
    """Validate and parse flat tool-schema edits into typed representations.

    Single source of truth for per-edit structural validation (shape,
    op constraints, field types) and anchor parsing.
    """
    result: list[HashlineEdit] = []
    for index, edit in enumerate(edits):
        _assert_edit_item(edit, index)

        op: str = edit["op"]
        if op == "replace":
            pos = parse_anchor_ref(edit["pos"])
            end = parse_anchor_ref(edit["end"]) if edit.get("end") else None
            lines = _hashline_parse_text(edit.get("lines"))
            result.append(ReplaceEdit(op="replace", pos=pos, end=end, lines=lines))
        elif op == "append":
            pos = parse_anchor_ref(edit["pos"]) if edit.get("pos") else None
            lines = _hashline_parse_text(edit.get("lines"))
            result.append(AppendEdit(op="append", pos=pos, lines=lines))
        elif op == "prepend":
            pos = parse_anchor_ref(edit["pos"]) if edit.get("pos") else None
            lines = _hashline_parse_text(edit.get("lines"))
            result.append(PrependEdit(op="prepend", pos=pos, lines=lines))
        elif op == "replace_text":
            result.append(
                ReplaceTextEdit(
                    op="replace_text",
                    oldText=_normalize_exact_text(edit["oldText"]),
                    newText=_normalize_exact_text(edit["newText"]),
                )
            )

    return result


def _normalize_exact_text(text: str | None) -> str:
    """Normalize line endings for exact text matching.

    Returns empty string for None/falsy input (matching TS behavior where
    undefined is caught by the ! assertion — callers always pass strings
    after _assert_edit_item validation).
    """
    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ─── Request Normalization ─────────────────────────────────────────────────
# Converges model dialects onto the canonical {path, edits: [{op, ...}]} shape.
# Ported from pi-hashline-edit/src/edit-normalize.ts.


def _coerce_edits_array(edits: object) -> object:
    """Parse edits when a model serializes it as a JSON string.

    Tries JSON first, then ast.literal_eval for Python-style literals
    (some models use single quotes in "code-like" fields).
    """
    if not isinstance(edits, str):
        return edits

    import json
    try:
        parsed = json.loads(edits)
        return parsed if isinstance(parsed, list) else edits
    except json.JSONDecodeError:
        pass

    # Fallback: Python literal (handles single quotes, Python-style syntax)
    import ast
    try:
        parsed = ast.literal_eval(edits)
        return parsed if isinstance(parsed, list) else edits
    except (ValueError, SyntaxError):
        return edits


_TOP_LEVEL_TEXT_REPLACE_KEYS = ("oldText", "newText", "old_text", "new_text")


def _backfill_edit_op(item: object) -> object:
    """Add op: 'replace_text' to edit items that have oldText/newText but no op."""
    if not isinstance(item, dict):
        return item
    if isinstance(item.get("op"), str):
        return item
    if isinstance(item.get("oldText"), str) and isinstance(item.get("newText"), str):
        return {"op": "replace_text", **item}
    return item


def normalize_edit_request(input: object) -> object:
    """Normalize a raw edit-tool request into the canonical hashline shape.

    Handles:
    - file_path → path alias
    - Top-level oldText/newText or old_text/new_text → edits[0] replace_text
    - edits-as-JSON-string → array
    - Missing op on text-replace edit items → "replace_text"
    """
    if not isinstance(input, dict):
        # Pass through non-dict payloads so downstream validation can reject
        # them with a precise error (matching TS edit-normalize.ts behavior).
        return input  # type: ignore[return-value]

    record = dict(input)

    # file_path → path alias
    if not isinstance(record.get("path"), str) and isinstance(record.get("file_path"), str):
        record["path"] = record.pop("file_path")

    # Validate top-level text replace aliases
    present_keys = [k for k in _TOP_LEVEL_TEXT_REPLACE_KEYS if k in record]
    if present_keys:
        for k in present_keys:
            if not isinstance(record[k], str):
                raise ValueError(f'Edit request field "{k}" must be a string.')

        has_camel = "oldText" in record or "newText" in record
        has_snake = "old_text" in record or "new_text" in record
        if has_camel and has_snake:
            raise ValueError(
                "Edit request cannot mix legacy camelCase and snake_case fields. "
                "Use either oldText/newText or old_text/new_text."
            )
        if has_camel and not ("oldText" in record and "newText" in record):
            raise ValueError("Legacy top-level replace requires both oldText and newText.")
        if has_snake and not ("old_text" in record and "new_text" in record):
            raise ValueError("Legacy top-level replace requires both old_text and new_text.")

    has_edits = "edits" in record

    # edits-as-JSON-string → array
    if has_edits:
        record["edits"] = _coerce_edits_array(record["edits"])

    existing_edits = record.get("edits") if isinstance(record.get("edits"), list) else None

    # Top-level native oldText/newText with no structured edits → fold into edits
    if not has_edits or (isinstance(existing_edits, list) and len(existing_edits) == 0):
        top_level: dict[str, str] | None = None
        if isinstance(record.get("oldText"), str) and isinstance(record.get("newText"), str):
            top_level = {"oldText": record["oldText"], "newText": record["newText"]}
        elif isinstance(record.get("old_text"), str) and isinstance(record.get("new_text"), str):
            top_level = {"oldText": record["old_text"], "newText": record["new_text"]}

        if top_level:
            # Strip top-level text-replace keys
            for k in _TOP_LEVEL_TEXT_REPLACE_KEYS:
                record.pop(k, None)
            return {**record, "edits": [{"op": "replace_text", **top_level}]}

    # Backfill missing op on edit items
    if isinstance(existing_edits, list):
        record["edits"] = [_backfill_edit_op(item) for item in existing_edits]

    return record
