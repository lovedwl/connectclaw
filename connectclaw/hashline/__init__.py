"""Hashline — hash-anchored line editing protocol.

Provides hash-anchored read and edit tools for AI coding agents.
Each line gets a short content hash so edits carry verifiable references
instead of raw text. Stale anchors are caught before they touch the file.

Inspired by oh-my-pi (github.com/can1357/oh-my-pi).
"""

from .apply import (
    ApplyResult,
    apply_hashline_edits,
    execute_edit_pipeline,
)
from .config import (
    HashlineConfig,
    get_hash_length,
    get_grep_enabled,
    reload_config,
)
from .format import (
    compute_affected_line_range,
    compute_changed_line_range,
    format_hashline_region,
)
from .guard import (
    clear_applied_payload,
    is_duplicate_applied_payload,
    record_applied_edit,
    record_noop_edit,
)
from .hash import (
    NIBBLE_STR,
    compute_line_hash,
    normalize_hash_input,
)
from .parse import (
    Anchor,
    AppendEdit,
    HashlineEdit,
    PrependEdit,
    ReplaceEdit,
    ReplaceTextEdit,
    normalize_edit_request,
    parse_anchor_ref,
    resolve_edit_anchors,
)
from .snapshot import (
    get_read_snapshot,
    get_read_snapshot_versions,
    remember_read_snapshot,
)

__all__ = [
    # Config
    "HashlineConfig",
    "get_hash_length",
    "get_grep_enabled",
    "reload_config",
    # Hash
    "NIBBLE_STR",
    "compute_line_hash",
    "normalize_hash_input",
    # Parse
    "Anchor",
    "HashlineEdit",
    "AppendEdit",
    "PrependEdit",
    "ReplaceEdit",
    "ReplaceTextEdit",
    "parse_anchor_ref",
    "resolve_edit_anchors",
    "normalize_edit_request",
    # Format
    "format_hashline_region",
    "compute_changed_line_range",
    "compute_affected_line_range",
    # Apply
    "ApplyResult",
    "apply_hashline_edits",
    "execute_edit_pipeline",
    # Snapshot
    "remember_read_snapshot",
    "get_read_snapshot",
    "get_read_snapshot_versions",
    # Guard
    "record_noop_edit",
    "record_applied_edit",
    "is_duplicate_applied_payload",
    "clear_applied_payload",
]
