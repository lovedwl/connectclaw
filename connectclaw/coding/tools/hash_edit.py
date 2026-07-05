"""Hashline edit tool — hash-anchored file modifications."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

import aiofiles

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.hashline.apply import execute_edit_pipeline
from connectclaw.hashline.diff_util import (
    detect_line_ending,
    generate_diff_string,
    has_mixed_line_endings,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from connectclaw.hashline.format import (
    compute_affected_line_range,
    format_hashline_region,
)
from connectclaw.hashline.guard import (
    clear_applied_payload,
    is_duplicate_applied_payload,
    record_applied_edit,
    record_noop_edit,
)
from connectclaw.hashline.parse import normalize_edit_request, resolve_edit_anchors
from connectclaw.hashline.snapshot import (
    get_read_snapshot,
    remember_read_snapshot,
)

# ─── Constants ──────────────────────────────────────────────────────────────

CHANGED_ANCHOR_TEXT_BUDGET_BYTES = 50 * 1024

# ─── Tool Description ───────────────────────────────────────────────────────

HASH_EDIT_DESCRIPTION = """Edit a file using hash-anchored line references from hash_read output.

Edits use LINE#HASH anchors from read output to target lines precisely.

Operations:
- replace: Replace one line (pos) or an inclusive range (pos + end) with lines
- append: Insert lines after pos; omit pos to append at EOF
- prepend: Insert lines before pos; omit pos to prepend at BOF
- replace_text: Replace an exact unique substring. Fails if not unique.

All edits in a single call validate against the same pre-edit snapshot and
apply bottom-up, so line numbers stay consistent across operations.

After a successful edit, the result includes fresh LINE#HASH anchors for
the changed region, usable directly in the next edit call on the same file
without a full re-read."""


class HashEditTool(AgentTool):
    name = "hash_edit"
    label = "Hash Edit"
    description = HASH_EDIT_DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit (relative or absolute)",
            },
            "edits": {
                "type": "array",
                "description": "List of edit operations to apply",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["replace", "append", "prepend", "replace_text"],
                            "description": "Edit operation type",
                        },
                        "pos": {
                            "type": "string",
                            "description": "Start anchor (LINE#HASH from read output)",
                        },
                        "end": {
                            "type": "string",
                            "description": "Inclusive end anchor for range replace",
                        },
                        "lines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Replacement content, one array entry per line, no LINE#HASH prefix",
                        },
                        "oldText": {
                            "type": "string",
                            "description": "Exact text to replace (replace_text only)",
                        },
                        "newText": {
                            "type": "string",
                            "description": "Replacement text (replace_text only)",
                        },
                    },
                    "required": ["op"],
                },
            },
        },
        "required": ["path", "edits"],
    }

    def __init__(self, cwd: str):
        self._cwd = cwd

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        raw_path = params["path"]
        raw_edits = params.get("edits", [])

        # ── Normalize request ──────────────────────────────
        try:
            normalized = normalize_edit_request({"path": raw_path, "edits": raw_edits})
        except ValueError as e:
            return AgentToolResult(
                content=[{"type": "text", "text": str(e)}],
                details={"error": str(e)},
            )

        path = normalized.get("path", raw_path)
        edits_raw = normalized.get("edits", raw_edits)
        absolute_path = self._resolve_path(path)

        if not os.path.isfile(absolute_path):
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error: File not found: {path}. Use the write tool to create new files."}],
            )

        if os.path.isdir(absolute_path):
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error: Path is a directory: {path}"}],
            )

        if not isinstance(edits_raw, list) or len(edits_raw) == 0:
            return AgentToolResult(
                content=[{"type": "text", "text": "Error: No edits provided."}],
            )

        # ── Read file ─────────────────────────────────────
        try:
            async with aiofiles.open(absolute_path, "r") as f:
                content = await f.read()
        except UnicodeDecodeError:
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": f"Error: File is binary or not UTF-8: {path}. Hashline edit only supports text files.",
                }],
            )
        except Exception as e:
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error reading file: {e}"}],
            )

        bom, raw_text = strip_bom(content)
        original_ending = detect_line_ending(raw_text)
        mixed_ending_warning = None
        if has_mixed_line_endings(raw_text):
            ending_label = "CRLF" if original_ending == "\r\n" else "LF"
            mixed_ending_warning = (
                f"File had mixed line endings (CRLF and LF); "
                f"this edit rewrote it uniformly as {ending_label}."
            )

        original_normalized = normalize_to_lf(raw_text)

        # ── Parse & validate edits ────────────────────────
        try:
            parsed_edits = resolve_edit_anchors(edits_raw)
        except ValueError as e:
            return AgentToolResult(
                content=[{"type": "text", "text": str(e)}],
            )

        # ── Duplicate-edit guard ──────────────────────────
        payload_key = json.dumps(edits_raw, sort_keys=True)
        if is_duplicate_applied_payload(absolute_path, payload_key):
            snapshot = get_read_snapshot(absolute_path)
            if snapshot is not None and snapshot == original_normalized:
                return AgentToolResult(
                    content=[{
                        "type": "text",
                        "text": (
                            f"[E_DUPLICATE_EDIT] This exact edit was already applied "
                            f"to {path} by your previous edit call — the file already "
                            f"contains this change. Do NOT resend the same payload: "
                            f"that would duplicate the inserted lines. Re-read the "
                            f"file to see the current state before editing again."
                        ),
                    }],
                )

        # ── Execute edit pipeline ─────────────────────────
        try:
            result, was_recovered = execute_edit_pipeline(
                absolute_path, parsed_edits, original_normalized
            )
        except ValueError as e:
            return AgentToolResult(
                content=[{"type": "text", "text": str(e)}],
                details={"error": str(e)},
            )

        # ── Handle noop ───────────────────────────────────
        if result.content == original_normalized:
            noop_count, escalate = record_noop_edit(absolute_path, payload_key)
            if escalate:
                return AgentToolResult(
                    content=[{
                        "type": "text",
                        "text": (
                            f"[E_NOOP_LOOP] Edit to {path} was a byte-identical "
                            f"no-op {noop_count} times in a row. STOP re-sending "
                            f"this payload. Re-read the file — the content you are "
                            f"trying to write already exists, or your anchors point "
                            f"at the wrong lines."
                        ),
                    }],
                )
            # Build noop response
            noop_detail = (
                "\n".join(
                    f"Edit {n.edit_index}: replacement for {n.loc} is identical "
                    f"to current content:\n  {n.loc}: {n.current_content}"
                    for n in result.noop_edits
                )
                if result.noop_edits
                else "The edits produced identical content."
            )
            warning_text = "\n\nWarnings:\n" + "\n".join(result.warnings) if result.warnings else ""
            return AgentToolResult(
                content=[{
                    "type": "text",
                    "text": f"No changes made to {path}\nClassification: noop\n{noop_detail}{warning_text}",
                }],
                details={
                    "diff": "",
                    "classification": "noop",
                    "warnings": result.warnings,
                },
            )

        # ── Build warnings ────────────────────────────────
        warnings: list[str] = list(result.warnings or [])
        if mixed_ending_warning:
            warnings.append(mixed_ending_warning)

        # ── Write file atomically ────────────────────────
        try:
            output_text = bom + restore_line_endings(result.content, original_ending)
            await _write_file_atomically(absolute_path, output_text)
        except Exception as e:
            return AgentToolResult(
                content=[{"type": "text", "text": f"Error writing file: {e}"}],
            )

        record_applied_edit(absolute_path, payload_key)

        # Update snapshot for chained edits
        remember_read_snapshot(absolute_path, result.content)

        # ── Build response ────────────────────────────────
        diff_str = generate_diff_string(original_normalized, result.content)

        # Anchor block for changed region
        result_lines = (
            result.content[:-1].split("\n")
            if result.content.endswith("\n")
            else result.content.split("\n")
        )
        anchor_range = compute_affected_line_range(
            result.first_changed_line,
            result.last_changed_line,
            len(result_lines),
        )
        if anchor_range:
            formatted = format_hashline_region(
                result_lines, anchor_range[0], anchor_range[1]
            )
            block = f"--- Anchors {anchor_range[0]}-{anchor_range[1]} ---\n{formatted}"
            if len(block.encode("utf-8")) > CHANGED_ANCHOR_TEXT_BUDGET_BYTES:
                anchors_text = "Anchors omitted; use hash_read for subsequent edits."
            else:
                anchors_text = block
        else:
            anchors_text = "Anchors omitted; use hash_read for subsequent edits."

        warning_block = "\n\nWarnings:\n" + "\n".join(warnings) if warnings else ""
        response_text = f"{anchors_text}{warning_block}"

        return AgentToolResult(
            content=[{"type": "text", "text": response_text}],
            details={
                "diff": diff_str,
                "first_changed_line": result.first_changed_line,
                "classification": "applied",
                "warnings": warnings,
            },
        )

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self._cwd, path))


async def _write_file_atomically(path: str, content: str) -> None:
    """Write content to a file atomically (temp file + rename)."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)

    tmp_path = os.path.join(dir_name, f".tmp-hashline-{uuid.uuid4().hex[:12]}")
    try:
        async with aiofiles.open(tmp_path, "w") as f:
            await f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise


def create_hash_edit_tool(cwd: str) -> HashEditTool:
    return HashEditTool(cwd)
