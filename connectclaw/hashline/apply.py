"""Apply engine — anchor validation, edit-span resolution, assembly.

Ported from pi-hashline-edit/src/hashline/apply.ts (MIT).

Three-phase pipeline:
  1. validate_anchor_edits — check hash matches, collect warnings + mismatches
  2. resolve_edit_spans   — map edits to character spans, dedup, conflict-detect, sort
  3. assemble_edit_result — apply spans back-to-front, compute changed range
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .format import compute_changed_line_range
from .hash import (
    RE_SIGNIFICANT,
    compute_hash_from_context,
    compute_line_hash,
    is_fuzzy_equivalent_line,
    normalize_hash_input,
)
from .parse import (
    Anchor,
    AppendEdit,
    HashlineEdit,
    PrependEdit,
    ReplaceEdit,
    ReplaceTextEdit,
    _get_bare_prefix_re,
)
from .snapshot import get_read_snapshot_versions
from .merge import three_way_merge


# ─── Types ──────────────────────────────────────────────────────────────────


@dataclass
class HashMismatch:
    line: int
    expected: str
    actual: str
    text_hint: str | None = None


@dataclass
class NoopEdit:
    edit_index: int
    loc: str
    current_content: str


@dataclass
class ApplyResult:
    content: str
    first_changed_line: int | None = None
    last_changed_line: int | None = None
    warnings: list[str] = field(default_factory=list)
    noop_edits: list[NoopEdit] = field(default_factory=list)


# ─── Mismatch Formatting ────────────────────────────────────────────────────

_CANDIDATE_TOTAL_LIMIT = 8
_CANDIDATE_PER_ANCHOR_LIMIT = 3


def _format_mismatch_error(
    mismatches: list[HashMismatch],
    file_lines: list[str],
    retry_lines: set[int] | None = None,
) -> str:
    """Build a descriptive [E_STALE_ANCHOR] error with current anchors."""
    if retry_lines is None:
        retry_lines = set()

    for m in mismatches:
        retry_lines.add(m.line)

    display_lines: set[int] = set()
    for m in mismatches:
        lo = max(1, m.line - 2)
        hi = min(len(file_lines), m.line + 2)
        for i in range(lo, hi + 1):
            display_lines.add(i)
    for line in retry_lines:
        display_lines.add(line)

    sorted_lines = sorted(display_lines)
    max_display = sorted_lines[-1] if sorted_lines else 1
    ln_width = len(str(max_display))
    stale_refs = ", ".join(f"{m.line}#{m.expected}" for m in mismatches)
    out: list[str] = [
        f"[E_STALE_ANCHOR] {len(mismatches)} 个过期锚点。"
        f"请使用下面的 >>> LINE#HASH 行重试；范围替换请保留两端锚点。",
        f"过期引用：{stale_refs}",
        "",
    ]

    prev = -1
    for num in sorted_lines:
        if prev != -1 and num > prev + 1:
            out.append("    ...")
        prev = num
        content = file_lines[num - 1]
        h = compute_line_hash(file_lines, num - 1)
        prefix = f"{str(num).rjust(ln_width)}#{h}"
        if num in retry_lines:
            out.append(f">>> {prefix}:{content}")
        else:
            out.append(f"    {prefix}:{content}")

    # Scan for fuzzy-match candidates
    hinted = [m for m in mismatches if m.text_hint is not None]
    if hinted:
        total_candidates = 0
        per_anchor: list[tuple[HashMismatch, dict]] = []

        for m in hinted:
            hint = m.text_hint
            assert hint is not None
            matches: list[int] = []
            for i in range(len(file_lines)):
                one_based = i + 1
                if one_based in display_lines:
                    continue
                if is_fuzzy_equivalent_line(hint, file_lines[i]):
                    matches.append(one_based)

            if total_candidates + len(matches) > _CANDIDATE_TOTAL_LIMIT:
                per_anchor.append((m, {"kind": "overflow", "count": len(matches)}))
            elif len(matches) > _CANDIDATE_PER_ANCHOR_LIMIT:
                total_candidates += len(matches)
                per_anchor.append((m, {"kind": "overflow", "count": len(matches)}))
            else:
                total_candidates += len(matches)
                per_anchor.append((m, {"kind": "list", "lines": matches}))

        has_any = any(
            r["kind"] == "overflow" or (r["kind"] == "list" and r["lines"])
            for _, r in per_anchor
        )
        if has_any:
            out.append("")
            out.append("你是不是想用（与内容匹配的过期锚点候选）：")
            for mismatch, result in per_anchor:
                if result["kind"] == "overflow":
                    out.append(
                        f"  为以下锚点找到 {result['count']} 条相似行："
                        f"{mismatch.line}#{mismatch.expected} — 重新读取以消除歧义"
                    )
                else:
                    for line_num in result["lines"]:
                        fresh_hash = compute_line_hash(file_lines, line_num - 1)
                        line_content = file_lines[line_num - 1]
                        out.append(
                            f"  {line_num}#{fresh_hash}:{line_content}"
                            f"   ← 对应过期锚点 {mismatch.line}#{mismatch.expected}"
                        )

    return "\n".join(out)


# ─── Line Index ─────────────────────────────────────────────────────────────


@dataclass
class LineIndex:
    file_lines: list[str]
    line_starts: list[int]
    has_terminal_newline: bool
    visible_line_count: int


def _build_line_index(content: str) -> LineIndex:
    file_lines = content.split("\n")
    line_starts: list[int] = []
    offset = 0
    for i, line in enumerate(file_lines):
        line_starts.append(offset)
        offset += len(line)
        if i < len(file_lines) - 1:
            offset += 1  # for the \n

    has_terminal_newline = content.endswith("\n")
    return LineIndex(
        file_lines=file_lines,
        line_starts=line_starts,
        has_terminal_newline=has_terminal_newline,
        visible_line_count=(
            len(file_lines) - 1 if has_terminal_newline else len(file_lines)
        ),
    )


# ─── Span Types ─────────────────────────────────────────────────────────────


@dataclass
class ReplaceSpan:
    kind: str = "replace"
    index: int = 0
    label: str = ""
    start: int = 0
    end: int = 0
    replacement: str = ""


@dataclass
class InsertSpan:
    kind: str = "insert"
    index: int = 0
    label: str = ""
    start: int = 0
    end: int = 0
    replacement: str = ""
    boundary: int | None = None
    insert_mode: str | None = None


ResolvedEditSpan = ReplaceSpan | InsertSpan


# ─── Edit Description ───────────────────────────────────────────────────────


def _preview_text(text: str) -> str:
    compact = text.replace("\n", "\\n")
    return compact[:29] + "..." if len(compact) > 32 else compact


def _describe_edit(edit: HashlineEdit) -> str:
    if isinstance(edit, ReplaceEdit):
        if edit.end:
            return f"替换 {edit.pos.line}#{edit.pos.hash}-{edit.end.line}#{edit.end.hash}"
        return f"替换 {edit.pos.line}#{edit.pos.hash}"
    elif isinstance(edit, AppendEdit):
        if edit.pos:
            return f"在 {edit.pos.line}#{edit.pos.hash} 之后追加"
        return "在文件末尾追加"
    elif isinstance(edit, PrependEdit):
        if edit.pos:
            return f"在 {edit.pos.line}#{edit.pos.hash} 之前插入"
        return "在文件开头插入"
    elif isinstance(edit, ReplaceTextEdit):
        return f'replace_text "{_preview_text(edit.oldText)}"'
    return "未知编辑"


# ─── Phase 1: Anchor Validation ─────────────────────────────────────────────


def _validate_anchor_edits(
    edits: list[HashlineEdit],
    line_index: LineIndex,
    warnings: list[str],
) -> tuple[list[HashMismatch], set[int]]:
    """Validate all anchor hashes against current file content.

    Returns (mismatches, retry_lines). Also appends boundary/single-anchor-range
    warnings to the shared warnings list.
    """
    mismatches: list[HashMismatch] = []
    retry_lines: set[int] = set()
    accepted_fuzzy_refs: set[str] = set()

    def _validate(ref: Anchor) -> bool:
        if ref.line < 1 or ref.line > len(line_index.file_lines):
            raise ValueError(
                f"[E_RANGE_OOB] 第 {ref.line} 行不存在"
                f"（文件共有 {line_index.visible_line_count} 行）"
            )
        line = line_index.file_lines[ref.line - 1]
        actual = compute_line_hash(line_index.file_lines, ref.line - 1)
        if actual == ref.hash:
            # Anti-collision guard: hash matches but text_hint differs → treat as stale
            if ref.text_hint is not None and not is_fuzzy_equivalent_line(ref.text_hint, line):
                mismatches.append(
                    HashMismatch(line=ref.line, expected=ref.hash, actual=actual, text_hint=ref.text_hint)
                )
                retry_lines.add(ref.line)
                return False
            return True

        if ref.text_hint is not None:
            # Forgiveness: recompute hash with the hint's content in current context
            prev_line = normalize_hash_input(
                line_index.file_lines[ref.line - 2] if ref.line > 1 else ""
            )
            next_line = normalize_hash_input(
                line_index.file_lines[ref.line] if ref.line < len(line_index.file_lines) else ""
            )
            hinted_hash = compute_hash_from_context(
                prev_line, normalize_hash_input(ref.text_hint), next_line
            )
            if hinted_hash == ref.hash and is_fuzzy_equivalent_line(ref.text_hint, line):
                key = f"{ref.line}:{ref.hash}:{ref.text_hint}"
                if key not in accepted_fuzzy_refs:
                    accepted_fuzzy_refs.add(key)
                    warnings.append(
                        f"已接受第 {ref.line} 行的模糊锚点校验："
                        f"精确哈希不匹配，但复制的行内容在空白/Unicode "
                        f"规范化后仍然匹配。"
                    )
                return True

        mismatches.append(
            HashMismatch(line=ref.line, expected=ref.hash, actual=actual, text_hint=ref.text_hint)
        )
        retry_lines.add(ref.line)
        return False

    for edit in edits:
        if isinstance(edit, ReplaceEdit):
            if edit.end:
                if edit.pos.line > edit.end.line:
                    raise ValueError(
                        f"[E_BAD_OP] 范围起始行 {edit.pos.line} 必须 "
                        f"<= 结束行 {edit.end.line}"
                    )
                start_ok = _validate(edit.pos)
                end_ok = _validate(edit.end)
                if not start_ok and end_ok:
                    retry_lines.add(edit.end.line)
                if start_ok and not end_ok:
                    retry_lines.add(edit.pos.line)
                if not start_ok or not end_ok:
                    continue
            elif not _validate(edit.pos):
                continue

            end_line = edit.end.line if edit.end else edit.pos.line
            if not edit.end and len(edit.lines) > 1:
                warnings.append(
                    f"在 {_describe_edit(edit)} 处的单锚点替换只替换了 "
                    f"第 {edit.pos.line} 行，但你提供了 {len(edit.lines)} 行替换 "
                    f'内容。如果你想替换一个范围，请添加 "end"。如果你想 '
                    f"把一行扩展为多行，请忽略此提示。"
                )

            # Boundary duplication warnings
            next_line = line_index.file_lines[end_line] if end_line < len(line_index.file_lines) else None
            if (
                next_line is not None
                and edit.lines
                and edit.lines[-1].strip()
                and RE_SIGNIFICANT.search(edit.lines[-1].strip())
                and edit.lines[-1].strip() == next_line.strip()
            ):
                warnings.append(
                    f"{_describe_edit(edit)} 之后可能存在边界重复："
                    f"替换内容的最后一行在去除空白后与后面保留下来的 "
                    f"行相同。"
                )

            prev_line = line_index.file_lines[edit.pos.line - 2] if edit.pos.line > 1 else None
            if (
                prev_line is not None
                and edit.lines
                and edit.lines[0].strip()
                and RE_SIGNIFICANT.search(edit.lines[0].strip())
                and edit.lines[0].strip() == prev_line.strip()
            ):
                warnings.append(
                    f"{_describe_edit(edit)} 之前可能存在边界重复："
                    f"替换内容的第一行在去除空白后与前面保留下来的 "
                    f"行相同。"
                )

        elif isinstance(edit, AppendEdit):
            if edit.pos and not _validate(edit.pos):
                continue
            if len(edit.lines) == 0:
                raise ValueError(
                    "[E_BAD_OP] append 操作的 lines 为空。"
                    "请提供要插入的内容，或移除该编辑。"
                )
            _warn_duplicate_insert("append", edit, line_index, warnings)

        elif isinstance(edit, PrependEdit):
            if edit.pos and not _validate(edit.pos):
                continue
            if len(edit.lines) == 0:
                raise ValueError(
                    "[E_BAD_OP] prepend 操作的 lines 为空。"
                    "请提供要插入的内容，或移除该编辑。"
                )
            _warn_duplicate_insert("prepend", edit, line_index, warnings)

        elif isinstance(edit, ReplaceTextEdit):
            pass  # No anchor validation needed

    return (mismatches, retry_lines)


def _warn_duplicate_insert(
    op: str,
    edit: AppendEdit | PrependEdit,
    line_index: LineIndex,
    warnings: list[str],
) -> None:
    """Warn when insert lines match the lines already adjacent at insertion point."""
    insert_lines = edit.lines
    n = len(insert_lines)
    if n == 0:
        return

    if op == "append":
        if edit.pos:
            compare_start = edit.pos.line
            compare_end = compare_start + n
        else:
            compare_start = line_index.visible_line_count - n
            compare_end = line_index.visible_line_count
    else:  # prepend
        if edit.pos:
            compare_end = edit.pos.line - 1
            compare_start = compare_end - n
        else:
            compare_start = 0
            compare_end = n

    if compare_start < 0 or compare_end > line_index.visible_line_count:
        return

    adjacent = line_index.file_lines[compare_start:compare_end]
    if len(adjacent) != n:
        return

    all_match = all(
        insert_lines[i].strip() == adjacent[i].strip() for i in range(n)
    )
    if not all_match:
        return

    has_significant = any(RE_SIGNIFICANT.search(line) for line in insert_lines)
    if not has_significant:
        return

    warnings.append(
        f"{_describe_edit(edit)} 处可能存在重复插入："
        f"插入的行与插入点旁边已有的行完全 "
        f"相同。如果之前的编辑调用已经应用了 "
        f"这次插入，请不要重复发送。"
    )


# ─── Content Preprocessing Warnings ─────────────────────────────────────────


def _warn_bare_hash_prefix_lines(
    edits: list[HashlineEdit],
    file_lines: list[str],
    warnings: list[str],
) -> None:
    """Warn when edit content may carry bare HH: hash prefixes."""
    bare_re = _get_bare_prefix_re()
    suspects: list[tuple[str, str]] = []

    for edit in edits:
        if isinstance(edit, ReplaceTextEdit):
            continue
        for line in edit.lines:
            m = bare_re.match(line)
            if m:
                suspects.append((line, m.group(1)))

    if not suspects:
        return

    file_hash_set = {compute_line_hash(file_lines, i) for i in range(len(file_lines))}
    match_count = sum(1 for _, h in suspects if h in file_hash_set)

    if match_count > 0 or len(suspects) >= 2:
        match_hint = (
            f" 有 {match_count} 个前缀与本文件中已有的行哈希匹配。"
            if match_count > 0
            else ""
        )
        warnings.append(
            f"{len(suspects)} 个编辑行以哈希和 ':' "
            f'开头（例如 {suspects[0][0]!r}）。{match_hint} 如果你是从 '
            f'"read" 输出中复制的，它们是哈希前缀而非文件内容 — 请将 '
            f'"lines" 作为字面内容重新发送。'
        )


def _maybe_warn_suspicious_unicode(edits: list[HashlineEdit], warnings: list[str]) -> None:
    """Warn per-edit when content contains literal \\uDDDD patterns.

    TS uses case-insensitive /\\uDDDD/i — we match the same by lowercasing.
    Each edit with suspicious content gets its own warning (matching TS behavior).
    """
    import re
    # Match literal backslash + u + 4 hex-ish digits (case-insensitive).
    # raw: \\ → regex \\ → matches single literal \
    _suspicious_re = re.compile(r"\\u[dD]{4}", re.IGNORECASE)
    for edit in edits:
        if isinstance(edit, ReplaceTextEdit):
            continue
        if any(_suspicious_re.search(line) for line in edit.lines):
            warnings.append(
                "在编辑内容中检测到字面量 \\uDDDD；未进行自动修正。 "
                "请确认这应该是一个真正的 Unicode 转义还是纯文本。"
            )


# ─── Phase 2: Edit Span Resolution ──────────────────────────────────────────


def _resolve_edit_to_span(
    edit: HashlineEdit,
    index: int,
    content: str,
    line_index: LineIndex,
    noop_edits: list[NoopEdit],
) -> ResolvedEditSpan | None:
    """Map a validated edit to a character-level span."""
    file_lines = line_index.file_lines
    line_starts = line_index.line_starts
    has_terminal_newline = line_index.has_terminal_newline

    if isinstance(edit, ReplaceEdit):
        start_line = edit.pos.line
        end_line = edit.end.line if edit.end else edit.pos.line
        original_lines = file_lines[start_line - 1 : end_line]

        if (
            len(original_lines) == len(edit.lines)
            and all(original_lines[i] == edit.lines[i] for i in range(len(edit.lines)))
        ):
            noop_edits.append(
                NoopEdit(
                    edit_index=index,
                    loc=f"{edit.pos.line}#{edit.pos.hash}",
                    current_content="\n".join(original_lines),
                )
            )
            return None

        if edit.lines:
            return ReplaceSpan(
                kind="replace",
                index=index,
                label=_describe_edit(edit),
                start=line_starts[start_line - 1],
                end=line_starts[end_line - 1] + len(file_lines[end_line - 1]),
                replacement="\n".join(edit.lines),
            )

        # Empty replacement
        if start_line == 1 and end_line == len(file_lines):
            return ReplaceSpan(
                kind="replace",
                index=index,
                label=_describe_edit(edit),
                start=0,
                end=len(content),
                replacement="",
            )

        if end_line < len(file_lines):
            return ReplaceSpan(
                kind="replace",
                index=index,
                label=_describe_edit(edit),
                start=line_starts[start_line - 1],
                end=line_starts[end_line],
                replacement="",
            )

        return ReplaceSpan(
            kind="replace",
            index=index,
            label=_describe_edit(edit),
            start=max(0, line_starts[start_line - 1] - 1),
            end=line_starts[end_line - 1] + len(file_lines[end_line - 1]),
            replacement="",
        )

    elif isinstance(edit, AppendEdit):
        inserted_text = "\n".join(edit.lines)
        if len(content) == 0:
            return InsertSpan(
                kind="insert",
                index=index,
                label=_describe_edit(edit),
                start=0,
                end=0,
                replacement=inserted_text,
                boundary=_compute_insertion_boundary(edit, line_index),
                insert_mode="append-empty-origin",
            )

        if not edit.pos:
            return InsertSpan(
                kind="insert",
                index=index,
                label=_describe_edit(edit),
                start=len(content),
                end=len(content),
                replacement=(
                    f"{inserted_text}\n" if has_terminal_newline else f"\n{inserted_text}"
                ),
                boundary=_compute_insertion_boundary(edit, line_index),
            )

        is_sentinel = has_terminal_newline and edit.pos.line == len(file_lines)
        return InsertSpan(
            kind="insert",
            index=index,
            label=_describe_edit(edit),
            start=(
                len(content)
                if is_sentinel
                else line_starts[edit.pos.line - 1] + len(file_lines[edit.pos.line - 1])
            ),
            end=(
                len(content)
                if is_sentinel
                else line_starts[edit.pos.line - 1] + len(file_lines[edit.pos.line - 1])
            ),
            replacement=f"{inserted_text}\n" if is_sentinel else f"\n{inserted_text}",
            boundary=_compute_insertion_boundary(edit, line_index),
        )

    elif isinstance(edit, PrependEdit):
        inserted_text = "\n".join(edit.lines)
        start = line_starts[edit.pos.line - 1] if edit.pos else 0
        return InsertSpan(
            kind="insert",
            index=index,
            label=_describe_edit(edit),
            start=start,
            end=start,
            replacement=(
                inserted_text if len(content) == 0 else f"{inserted_text}\n"
            ),
            boundary=_compute_insertion_boundary(edit, line_index),
            insert_mode=("prepend-empty-origin" if len(content) == 0 else None),
        )

    elif isinstance(edit, ReplaceTextEdit):
        match = _find_exact_unique_text_match(content, edit.oldText)
        if edit.oldText == edit.newText:
            noop_edits.append(
                NoopEdit(
                    edit_index=index,
                    loc=f'replace_text "{_preview_text(edit.oldText)}"',
                    current_content=edit.oldText,
                )
            )
            return None
        return ReplaceSpan(
            kind="replace",
            index=index,
            label=_describe_edit(edit),
            start=match["start"],
            end=match["end"],
            replacement=edit.newText,
        )

    return None


def _compute_insertion_boundary(
    edit: AppendEdit | PrependEdit,
    line_index: LineIndex,
) -> int:
    if isinstance(edit, PrependEdit):
        return edit.pos.line - 1 if edit.pos else 0
    # append
    if not edit.pos:
        return line_index.visible_line_count
    if line_index.has_terminal_newline and edit.pos.line == len(line_index.file_lines):
        return line_index.visible_line_count
    return edit.pos.line


def _find_exact_unique_text_match(
    content: str,
    old_text: str,
) -> dict:
    if not old_text:
        raise ValueError("[E_BAD_OP] replace_text 需要非空的 oldText。")

    matches: list[int] = []
    from_idx = 0
    while from_idx <= len(content) - len(old_text):
        idx = content.find(old_text, from_idx)
        if idx == -1:
            break
        matches.append(idx)
        from_idx = idx + 1

    for j in range(1, len(matches)):
        if matches[j] - matches[j - 1] < len(old_text):
            raise ValueError(
                "[E_MULTI_MATCH] replace_text 找到重叠的精确匹配；"
                "请重新读取并使用 hashline 编辑。"
            )

    if not matches:
        raise ValueError(
            "[E_NO_MATCH] replace_text 在当前文件中找不到精确且唯一的匹配。"
        )

    if len(matches) > 1:
        raise ValueError(
            "[E_MULTI_MATCH] replace_text 在当前文件中找到多个精确匹配。"
            "请重新读取并使用 hashline 编辑。"
        )

    return {"start": matches[0], "end": matches[0] + len(old_text)}


def _assert_no_conflicting_spans(spans: list[ResolvedEditSpan]) -> None:
    for left_i in range(len(spans)):
        left = spans[left_i]
        for right_i in range(left_i + 1, len(spans)):
            right = spans[right_i]

            if isinstance(left, InsertSpan) and isinstance(right, InsertSpan):
                if left.boundary == right.boundary:
                    raise ValueError(
                        f"[E_EDIT_CONFLICT] 单个请求中存在冲突的编辑："
                        f"编辑 {left.index}（{left.label}）与编辑 {right.index} "
                        f"（{right.label}）指向同一插入边界。"
                        f"请将它们合并为一个不重叠的修改，或拆分请求。"
                    )
                continue

            if left.kind == "replace" and right.kind == "replace":
                if left.start < right.end and right.start < left.end:
                    raise ValueError(
                        f"[E_EDIT_CONFLICT] 单个请求中存在冲突的编辑："
                        f"编辑 {left.index}（{left.label}）与编辑 {right.index} "
                        f"（{right.label}）在同一原始行范围上重叠。"
                        f"请将它们合并为一个不重叠的修改，或拆分请求。"
                    )
                continue

            # One replace, one insert
            rep = left if left.kind == "replace" else right
            ins = left if left.kind == "insert" else right
            if ins.start >= rep.start and ins.start < rep.end:
                raise ValueError(
                    f"[E_EDIT_CONFLICT] 单个请求中存在冲突的编辑："
                    f"编辑 {left.index}（{left.label}）与编辑 {right.index} "
                    f"（{right.label}）无法一起应用，因为其中一个插入到了 "
                    f"被替换的原始范围内。请合并它们或拆分请求。"
                )


def _resolve_edit_spans(
    edits: list[HashlineEdit],
    content: str,
    line_index: LineIndex,
    noop_edits: list[NoopEdit],
) -> list[ResolvedEditSpan]:
    seen_span_keys: set[str] = set()
    resolved: list[ResolvedEditSpan] = []

    for idx, edit in enumerate(edits):
        span = _resolve_edit_to_span(edit, idx, content, line_index, noop_edits)
        if span is None:
            continue

        if isinstance(span, InsertSpan):
            span_key = f"insert:{span.boundary}:{span.replacement}"
        else:
            span_key = f"replace:{span.start}:{span.end}:{span.replacement}"
        if span_key in seen_span_keys:
            continue
        seen_span_keys.add(span_key)
        resolved.append(span)

    _assert_no_conflicting_spans(resolved)

    # Sort back-to-front for safe in-place assembly
    return sorted(
        resolved,
        key=lambda s: (
            -s.end,
            0 if isinstance(s, ReplaceSpan) else 1,
            -(s.boundary or -1) if isinstance(s, InsertSpan) else 0,
            s.index,  # ascending by index (TS: left.index - right.index)
        ),
    )


# ─── Phase 3: Assembly ─────────────────────────────────────────────────────


def _assemble_edit_result(
    content: str,
    spans: list[ResolvedEditSpan],
) -> str:
    """Apply ordered spans to content in reverse (back-to-front) order."""
    result = content
    for span in spans:
        replacement = span.replacement
        if isinstance(span, InsertSpan):
            if span.insert_mode == "append-empty-origin":
                replacement = replacement if len(result) == 0 else f"\n{replacement}"
            elif span.insert_mode == "prepend-empty-origin":
                replacement = replacement if len(result) == 0 else f"{replacement}\n"
        result = result[: span.start] + replacement + result[span.end :]
    return result


def _assert_does_not_empty_file(original: str, result: str) -> None:
    if len(original) > 0 and len(result) == 0:
        raise ValueError(
            "[E_WOULD_EMPTY] 拒绝通过编辑清空一个非空文件。"
            "如果是有意为之，请使用 write 工具或 bash。"
        )


# ─── Top-level Entry Point ─────────────────────────────────────────────────


def apply_hashline_edits(
    content: str,
    edits: list[HashlineEdit],
) -> ApplyResult:
    """Apply hashline-anchored edits to file content.

    Three-phase pipeline:
      1. validate_anchor_edits — check hash matches
      2. resolve_edit_spans   — map to character spans
      3. assemble_edit_result — apply back-to-front
    """
    if not edits:
        return ApplyResult(content=content)

    line_index = _build_line_index(content)
    noop_edits: list[NoopEdit] = []
    warnings: list[str] = []

    # Phase 1: validate anchors
    mismatches, retry_lines = _validate_anchor_edits(edits, line_index, warnings)
    if mismatches:
        raise ValueError(
            _format_mismatch_error(mismatches, line_index.file_lines, retry_lines)
        )

    _warn_bare_hash_prefix_lines(edits, line_index.file_lines, warnings)
    _maybe_warn_suspicious_unicode(edits, warnings)

    # Phase 2: resolve edits to ordered spans
    ordered_spans = _resolve_edit_spans(edits, content, line_index, noop_edits)

    # Phase 3: assemble result
    result = _assemble_edit_result(content, ordered_spans)
    _assert_does_not_empty_file(content, result)

    changed_range = compute_changed_line_range(content, result)

    return ApplyResult(
        content=result,
        first_changed_line=changed_range[0] if changed_range else None,
        last_changed_line=changed_range[1] if changed_range else None,
        warnings=warnings,
        noop_edits=noop_edits,
    )


# ─── Full edit pipeline with snapshot recovery ──────────────────────────────


def execute_edit_pipeline(
    path: str,
    edits: list[HashlineEdit],
    current_content: str,
) -> tuple[ApplyResult, bool]:
    """Execute the edit pipeline with snapshot-based stale-anchor recovery.

    Returns (result, was_recovered).
    Raises ValueError on failure (with fresh anchors for retry).
    """
    try:
        result = apply_hashline_edits(current_content, edits)
        return (result, False)
    except ValueError as e:
        msg = str(e)
        if not msg.startswith("[E_STALE_ANCHOR]"):
            raise

        # Attempt snapshot recovery
        versions = get_read_snapshot_versions(path)
        # Filter out versions identical to current content
        versions = [v for v in versions if v != current_content]

        if not versions:
            raise

        any_anchor_valid = False
        for snapshot in versions:
            try:
                snapshot_result = apply_hashline_edits(snapshot, edits)
            except ValueError:
                continue

            any_anchor_valid = True
            merged = three_way_merge(snapshot, snapshot_result.content, current_content)
            if merged is None:
                continue

            # Recompute changed range against live file
            changed_range = compute_changed_line_range(current_content, merged)
            result = ApplyResult(
                content=merged,
                first_changed_line=changed_range[0] if changed_range else None,
                last_changed_line=changed_range[1] if changed_range else None,
                warnings=[
                    "已通过将本次编辑在最近一次读取的文件版本上重放，并合并到当前内容"
                    "（精确合并，不进行位置迁移），从而恢复了过期锚点。"
                    "请检查差异以确认结果。",
                    *(snapshot_result.warnings or []),
                ],
                noop_edits=snapshot_result.noop_edits,
            )
            return (result, True)

        if any_anchor_valid:
            suffix = (
                "\n（已尝试恢复：你的锚点匹配该文件较早的一次读取，"
                "但重放该编辑与之后的改动发生冲突。"
                "请重新读取以获取当前锚点。）"
            )
        else:
            suffix = (
                "\n（你的锚点不匹配该文件任何一次最近的读取 — "
                "它们可能来自过期的上下文或复制有误。"
                "编辑前请重新读取。）"
            )
        raise ValueError(msg + suffix)
