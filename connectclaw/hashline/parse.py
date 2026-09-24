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
        return f'[E_BAD_REF] 无效的行引用 "{ref}"。应为 "LINE#HASH"（例如 "{example}"）。'
    if re.match(r"^\d+\s*$", core):
        return f'[E_BAD_REF] 无效的行引用 "{ref}"：缺少哈希，请使用 read 输出中的 "LINE#HASH"（例如 "{example}"）。'
    if re.match(r"^\d+\s*:", core):
        return f'[E_BAD_REF] 无效的行引用 "{ref}"：分隔符错误，请使用 "LINE#HASH" 而不是 "LINE:..."。'

    hash_match = re.match(r"^(\d+)\s*#\s*([^\s:]+)(?:\s*:.*)?$", core)
    if hash_match:
        line = int(hash_match.group(1))
        hash_str = hash_match.group(2)
        if line < 1:
            return f'[E_BAD_REF] 行号必须 >= 1，但在 "{ref}" 中得到 {line}。'
        if len(hash_str) != config_len:
            if (
                HASH_ALPHABET_RE.match(hash_str)
                and HASH_LENGTH_MIN <= len(hash_str) <= HASH_LENGTH_MAX
            ):
                return f'[E_BAD_REF] 无效的行引用 "{ref}"：本会话的哈希长度为 {config_len}，但该锚点有 {len(hash_str)} 个字符 — 它看起来来自过期的上下文或不同的配置。请重新读取文件以获取当前锚点。'
            return f'[E_BAD_REF] 无效的行引用 "{ref}"：哈希必须正好是来自 {NIBBLE_STR} 的 {config_len} 个字符（例如 "{example}"）。'
        if not HASH_ALPHABET_RE.match(hash_str):
            return f'[E_BAD_REF] 无效的行引用 "{ref}"：哈希包含无效字符，哈希只使用字母表 {NIBBLE_STR}。'

    missing_hash_match = re.match(r"^(\d+)\s*#\s*$", core)
    if missing_hash_match:
        return f'[E_BAD_REF] 无效的行引用 "{ref}"：在 "#" 之后缺少哈希，请使用 read 输出中的 "LINE#HASH"。'

    if re.match(r"^0+\s*#", core):
        return f'[E_BAD_REF] 行号必须 >= 1，但在 "{ref}" 中得到 0。'

    return f'[E_BAD_REF] 无效的行引用 "{trimmed or ref}"。应为 "LINE#HASH"（例如 "{example}"）。'


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
        raise ValueError(f'[E_BAD_REF] 行号必须 >= 1，但在 "{ref}" 中得到 {line}。')

    hash_str = match.group(2)
    config_len = get_hash_length()
    if len(hash_str) != config_len:
        if (
            HASH_ALPHABET_RE.match(hash_str)
            and HASH_LENGTH_MIN <= len(hash_str) <= HASH_LENGTH_MAX
        ):
            raise ValueError(
                f'[E_BAD_REF] 无效的行引用 "{ref}"：本会话的哈希长度为 {config_len} '
                f"但该锚点有 {len(hash_str)} 个字符 — "
                f"它看起来来自过期的上下文或不同的配置。"
                f"请重新读取文件以获取当前锚点。"
            )
        raise ValueError(
            f'[E_BAD_REF] 无效的行引用 "{ref}"：哈希必须正好是 '
            f'来自 {NIBBLE_STR} 的 {config_len} 个字符（例如 "{_example_anchor()}"）。'
        )

    if not HASH_ALPHABET_RE.match(hash_str):
        raise ValueError(
            f'[E_BAD_REF] 无效的行引用 "{ref}"：哈希包含无效 '
            f"字符，哈希只使用字母表 {NIBBLE_STR}。"
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
                f'[E_INVALID_PATCH] "lines" 必须包含字面文件内容，'
                f'而不是渲染后的 "LINE#HASH:" 或 diff "+/-" 前缀。'
                f"有问题的行：{line!r}"
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
            f"编辑 {index} 包含未知或不支持的字段：{', '.join(sorted(unknown_keys))}。"
        )

    op = edit.get("op")
    if not isinstance(op, str):
        raise ValueError(f'编辑 {index} 需要 "op" 字符串。')
    if op not in ("replace", "append", "prepend", "replace_text"):
        raise ValueError(
            f'[E_BAD_OP] 编辑 {index} 使用了未知的 op "{op}"。'
            f'应为 "replace"、"append"、"prepend" 或 "replace_text"。'
        )

    if "pos" in edit and not isinstance(edit["pos"], str):
        raise ValueError(f'编辑 {index} 的字段 "pos" 提供时必须是字符串。')
    if "end" in edit and not isinstance(edit.get("end", ""), str):
        raise ValueError(f'编辑 {index} 的字段 "end" 提供时必须是字符串。')
    if "oldText" in edit and not isinstance(edit["oldText"], str):
        raise ValueError(f'编辑 {index} 的字段 "oldText" 提供时必须是字符串。')
    if "newText" in edit and not isinstance(edit.get("newText", ""), str):
        raise ValueError(f'编辑 {index} 的字段 "newText" 提供时必须是字符串。')
    if "lines" in edit and not (
        isinstance(edit["lines"], list)
        and all(isinstance(item, str) for item in edit["lines"])
    ):
        raise ValueError(f'编辑 {index} 的字段 "lines" 必须是字符串数组。')

    if op == "replace_text":
        if not isinstance(edit.get("oldText"), str) or not isinstance(edit.get("newText"), str):
            raise ValueError(
                f'[E_BAD_OP] 使用 op "replace_text" 的编辑 {index} 需要 '
                f'字符串 "oldText" 和 "newText" 字段。'
            )
        if "pos" in edit or "end" in edit or "lines" in edit:
            raise ValueError(
                f'使用 op "replace_text" 的编辑 {index} 只支持 "oldText" 和 "newText"。'
            )
        return

    if "lines" not in edit:
        raise ValueError(f'编辑 {index} 需要一个 "lines" 字段。')

    if "oldText" in edit or "newText" in edit:
        raise ValueError(
            f'使用 op "{op}" 的编辑 {index} 不支持 "oldText" 或 "newText"。'
        )

    if op == "replace" and not isinstance(edit.get("pos"), str):
        raise ValueError(
            f'[E_BAD_OP] 使用 op "replace" 的编辑 {index} 需要一个 "pos" 锚点字符串。'
        )

    if op in ("append", "prepend") and "end" in edit:
        raise ValueError(
            f'[E_BAD_OP] 使用 op "{op}" 的编辑 {index} 不支持 "end"。'
            f'请使用 "pos"，或省略它以在文件边界插入。'
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
                raise ValueError(f'编辑请求字段 "{k}" 必须是字符串。')

        has_camel = "oldText" in record or "newText" in record
        has_snake = "old_text" in record or "new_text" in record
        if has_camel and has_snake:
            raise ValueError(
                "编辑请求不能混用旧的 camelCase 和 snake_case 字段。"
                "请使用 oldText/newText 或 old_text/new_text 其中之一。"
            )
        if has_camel and not ("oldText" in record and "newText" in record):
            raise ValueError("旧的顶层替换需要同时提供 oldText 和 newText。")
        if has_snake and not ("old_text" in record and "new_text" in record):
            raise ValueError("旧的顶层替换需要同时提供 old_text 和 new_text。")

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
