"""Part 2: guard/snapshot/format/diff/edge cases reference."""
import json
import sys

sys.path.insert(0, ".")

from connectclaw.hashline.hash import compute_line_hash, compute_hash_from_context
from connectclaw.hashline.parse import resolve_edit_anchors
from connectclaw.hashline.apply import apply_hashline_edits, execute_edit_pipeline
from connectclaw.hashline.format import (
    compute_affected_line_range, compute_changed_line_range, format_hashline_region,
)
from connectclaw.hashline.snapshot import (
    remember_read_snapshot, get_read_snapshot, get_read_snapshot_versions,
    reset_read_snapshots,
)
from connectclaw.hashline.guard import (
    record_noop_edit, record_applied_edit, is_duplicate_applied_payload,
    clear_applied_payload, reset_noop_guard,
)
from connectclaw.hashline.diff_util import (
    detect_line_ending, normalize_to_lf, restore_line_endings,
    has_mixed_line_endings, strip_bom, generate_diff_with_hashes,
    generate_diff_string,
)
import connectclaw.hashline.snapshot as snap_mod
import connectclaw.hashline.guard as guard_mod

out = []
emit = lambda *a: out.append(" ".join(repr(x) for x in a))

# ── guard ─────────────────────────────────────────────
reset_noop_guard()
emit("noop1", record_noop_edit("/a", "k1"))   # (1, False)
emit("noop2", record_noop_edit("/a", "k1"))   # (2, False)
emit("noop3", record_noop_edit("/a", "k1"))   # (3, True)
emit("noop4", record_noop_edit("/a", "k2"))   # (1, False) different key resets
emit("noop5", record_noop_edit("/a", "k1"))   # (1, False) key changed back
emit("dupe1", is_duplicate_applied_payload("/a", "p1"))  # False
record_applied_edit("/a", "p1")
emit("dupe2", is_duplicate_applied_payload("/a", "p1"))  # True
emit("dupe3", is_duplicate_applied_payload("/a", "p2"))  # False
clear_applied_payload("/a")
emit("dupe4", is_duplicate_applied_payload("/a", "p1"))  # False
# record_applied_edit clears noop counter
record_noop_edit("/b", "k1")
record_noop_edit("/b", "k1")
record_applied_edit("/b", "p9")
emit("noop_after_applied", record_noop_edit("/b", "k1"))  # (1, False)
emit("guard_state_less", len(guard_mod._noop_tracker), len(guard_mod._applied_payload_tracker))

# ── snapshot ──────────────────────────────────────────
reset_read_snapshots()
remember_read_snapshot("/p", "v1")
emit("snap1", get_read_snapshot("/p"), get_read_snapshot_versions("/p"))
remember_read_snapshot("/p", "v1")
emit("snap_fusion", get_read_snapshot_versions("/p"))   # unchanged
remember_read_snapshot("/p", "v2")
remember_read_snapshot("/p", "v3")
emit("snap2", get_read_snapshot_versions("/p"))
via = snap_mod
emit("snap_state", via._path_order, {k: list(v) for k, v in via._path_map.items()})
# MRU eviction across paths
remember_read_snapshot("/q", "q1")
remember_read_snapshot("/r", "r1")
emit("snap3", via._path_order)
# touch /p again → promote
remember_read_snapshot("/p", "v4")
emit("snap_promote", via._path_order, {k: list(v) for k, v in via._path_map.items()})
# exceed versions per path
for i in range(5, 9):
    remember_read_snapshot("/p", f"v{i}")
emit("snap_versions_cap", get_read_snapshot_versions("/p"))
emit("snap_newest", get_read_snapshot("/p"))
# different content on /q bumps order
remember_read_snapshot("/q", "q2")
emit("snap_after_q", via._path_order)
emit("snap_recent", via._path_map["/q"])

# byte budget: force eviction with huge content (MAX_TOTAL_CHARS = 32MiB)
big = "x" * (4 * 1024 * 1024)
reset_read_snapshots()
remember_read_snapshot("/big1", big)
remember_read_snapshot("/big2", big)
remember_read_snapshot("/big3", big)
remember_read_snapshot("/big4", big)
remember_read_snapshot("/big5", big)
emit("snap_budget_paths", via._path_order)
emit("snap_budget_has_big1", via._path_map.get("/big1"))

# ── format ────────────────────────────────────────────
emit("aff1", compute_affected_line_range(3, 5, 10))
emit("aff2", compute_affected_line_range(None, None, 10))
emit("aff3", compute_affected_line_range(1, 1, 10, 0, 1))
emit("aff4", compute_affected_line_range(1, 20, 20))
emit("aff5", compute_affected_line_range(1, 1, 1, 2, 12))
emit("aff6", compute_affected_line_range(9, 10, 10, 2, 6))
fmt_lines = ["one", "two", "three", "four", "five"]
emit("fmt1", repr(format_hashline_region(fmt_lines, 1, 5)))
emit("fmt2", repr(format_hashline_region(fmt_lines, 2, 4)))
# changed range
emit("chr1", compute_changed_line_range("a\nb\nc", "a\nB\nc"))
emit("chr2", compute_changed_line_range("a\nb\nc", "a\nb\nc"))
emit("chr3", compute_changed_line_range("", "a\nb"))
emit("chr4", compute_changed_line_range("a\nb\nc\n", "a\nb\nc\nd\n"))
emit("chr5", compute_changed_line_range("a\nb\nc\n", "z\nb\nc\n"))
emit("chr6", compute_changed_line_range("a\nb\nc\n", "a\nb\n"))
emit("chr7", compute_changed_line_range("x", "y"))
emit("chr8", compute_changed_line_range("x\ny\nz", ""))
emit("chr9", compute_changed_line_range("aaaa", "aaab"))
emit("chr10", compute_changed_line_range("a\nb", "a\nb\nc\nd"))
emit("chr11", compute_changed_line_range("prefix\nfull", "prefixed\nfull"))

# ── more apply: warnings ──────────────────────────────
CONTENT = "alpha\nbeta\ngamma\ndelta\n"
cl = CONTENT.split("\n")[:-1]
def A(content, edits, label):
    try:
        r = apply_hashline_edits(content, resolve_edit_anchors(edits))
        return ("OK", repr(r.content), r.first_changed_line, r.last_changed_line, r.warnings, [(n.edit_index, n.loc, n.current_content) for n in r.noop_edits])
    except ValueError as e:
        return ("ERR", str(e))
def anchor_for_line(content, lineno):
    lines = content.split("\n")
    if content.endswith("\n"):
        lines = lines[:-1]
    return f"{lineno}#{compute_line_hash(lines, lineno - 1)}"

# single-anchor multi-line replace warning
emit("warn_single_multi", A(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 2), "lines": ["b1", "b2"]}], "x"))
# boundary duplication next
emit("warn_boundary_next", A(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 2), "lines": ["BETA", "gamma"]}], "x"))
# boundary duplication prev
emit("warn_boundary_prev", A(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 2), "lines": ["alpha", "BETA"]}], "x"))
# duplicate insert warning (append with same lines adjacent)
emit("warn_dup_insert_append", A(CONTENT, [{"op": "append", "pos": anchor_for_line(CONTENT, 1), "lines": ["beta", "gamma"]}], "x"))
emit("warn_dup_insert_prepend", A(CONTENT, [{"op": "prepend", "pos": anchor_for_line(CONTENT, 3), "lines": ["alpha", "beta"]}], "x"))
# bare hash prefix warning: line starts with hash equal to file hash + colon
h2 = compute_line_hash(cl, 1)
emit("warn_bare_prefix", A(CONTENT, [{"op": "append", "pos": anchor_for_line(CONTENT, 1), "lines": [f"{h2}:literal?"]}], "x"))
# bare hash prefix warning (two suspects but no hash match)
emit("warn_bare_prefix2", A(CONTENT, [{"op": "append", "lines": ["WW:foo", "XX:bar"]}], "x"))
# suspicious unicode warning
emit("warn_suspicious", A(CONTENT, [{"op": "append", "lines": [r"text \uDDDD more"]}], "x"))
# fuzzy acceptance (forgiveness) warning
# line 2 is "beta" ; provide hash computed from hint "beta " won't... construct: ref.hash = hash(ctx, "beta", ...) with hint beta but the line content is "beta " (trailing space) so actual hash differs
C_SPACED = "alpha\nbeta \ngamma\ndelta\n"
spaced_lines = C_SPACED.split("\n")[:-1]
prev = sp = "beta"
hint_hash = compute_hash_from_context("alpha", sp, "gamma")
emit("forgive_apply", A(C_SPACED, [{"op": "replace", "pos": f"2#{hint_hash}:beta", "lines": ["XX"]}], "x"))
# mismatch with candidate list (hinted mismatch where hint matches another line outside display)
# Make hint "gamma" but line 2 changed to BETA; hint gamma fuzzy-matches real gamma at line 3 (which is inside display range? line3 is adjacent so in display lines → skipped)
C_M = "alpha\nBETA\ngamma\ndelta\n"
emit("cand_inside", A(C_M, [{"op": "replace", "pos": f"2#{compute_line_hash(['alpha','beta','gamma','delta'],1)}:beta changed", "lines": ["X"]}], "x"))
# bigger file: candidate appears far away (outside display)
big_lines = [f"line{i}" for i in range(20)]
BIG = "\n".join(big_lines) + "\n"
# change line 5
BIG_EDITED_LINES = list(big_lines)
BIG_EDITED_LINES[4] = "CHANGED"
BIG_EDITED = "\n".join(BIG_EDITED_LINES) + "\n"
# anchor old line5 hash with hint "line17" (far away) → candidate list
old_hash5 = compute_line_hash(big_lines, 4)
emit("cand_far", A(BIG_EDITED, [{"op": "replace", "pos": f"5#{old_hash5}:line17", "lines": ["X"]}], "x"))
# candidate overflow (>3 matches, hint matching only equal lines—use common word)
EIGHT = "\n".join(["same"] * 10) + "\n"
emit("cand_overflow", A(EIGHT.replace("same", "same", 1), [{"op": "replace", "pos": "1#AZ:same", "lines": ["X"]}], "x"))
# replace_text CRLF normalization in oldText
emit("rt_crlf", A("a\r\nb\r\nc\r\n", [{"op": "replace_text", "oldText": "a\r\nb", "newText": "AB"}], "x"))
emit("rt_norm_identity", A("a\r\nb\r\nc\r\n", [{"op": "replace_text", "oldText": "a\r\nb\n", "newText": "AB"}], "x"))

# empty lines replace to empty at very end with no trailing newline
emit("empty_last_no_trail", A("alpha\nbeta", [{"op": "replace", "pos": anchor_for_line("alpha\nbeta", 2), "lines": []}], "x"))

# ── execute_edit_pipeline with snapshot recovery ─────
reset_read_snapshots()
path = "/tmp/mypath.py"
current = "aaa\nbbb\nccc\n"
# snapshot old version: old line 2 "bbb" replaced by "BBB"
old = "aaa\nbbb\nccc\n"
remember_read_snapshot(path, old)
# edit targets old anchor 2#(hash of bbb) replace with "NEW" ; live content differs
nl = old.split("\n")[:-1]
old_hash2 = compute_line_hash(nl, 1)
edits = resolve_edit_anchors([{"op": "replace", "pos": f"2#{old_hash2}", "lines": ["NEW"]}])
current = "aaa\nBBB\nccc\nddd\n"
try:
    r, recovered = execute_edit_pipeline(path, edits, current)
    emit("pipeline", recovered, repr(r.content), r.first_changed_line, r.last_changed_line, r.warnings, [(n.edit_index, n.loc, n.current_content) for n in r.noop_edits])
except ValueError as e:
    emit("pipeline_ERR", str(e))

# recovery conflict (merge returns None because patch doesn't apply)
old2 = "1\n2\n3\n4\n"
late = "1\n2\n3\n4\n"  # current where patch to current == current -> merged==current -> None
remember_read_snapshot(path, old2)
edits2 = resolve_edit_anchors([{"op": "replace", "pos": "2#" + compute_line_hash(old2.split("\n")[:-1], 1), "lines": ["2"]}])  # noop edit on snapshot
current2 = "1\n2\n3\n4\n"
try:
    r, recovered = execute_edit_pipeline(path, edits2, current2)
    emit("pipeline_noop_or_conflict", recovered, repr(r.content), r.warnings)
except ValueError as e:
    emit("pipeline_ERR2", str(e))

# non stale-anchor error passes through
try:
    r, recovered = execute_edit_pipeline(path, resolve_edit_anchors([{"op": "replace_text", "oldText": "nope", "newText": "x"}]), current)
    emit("pipeline_nonstale_ok", recovered, repr(r.content))
except ValueError as e:
    emit("pipeline_nonstale_ERR", str(e))

# no snapshots → passthrough raise
reset_read_snapshots()
try:
    r, recovered = execute_edit_pipeline("/nope.py", resolve_edit_anchors([{"op": "replace", "pos": "1#AB", "lines": ["x"]}]), "abc\n")
    emit("pipeline_nosnap_ok", repr(r.content))
except ValueError as e:
    emit("pipeline_nosnap_ERR", str(e)[:60], str(e)[-60:])

# ── diff_util ──────────────────────────────────────────
emit("dle1", detect_line_ending("a\r\nb\r\nc"))
emit("dle2", detect_line_ending("a\nb\nc"))
emit("dle3", detect_line_ending("a\r\nb\nc"))
emit("dle4", detect_line_ending(""))
emit("nl1", repr(normalize_to_lf("a\r\nb\rc\nd")))
emit("rl1", repr(restore_line_endings("a\nb", "\r\n")))
emit("rl2", repr(restore_line_endings("a\nb", "\n")))
emit("mixed1", has_mixed_line_endings("a\r\nb\r\n"))
emit("mixed2", has_mixed_line_endings("a\nb\n"))
emit("mixed3", has_mixed_line_endings("a\r\nb\n"))
emit("mixed4", has_mixed_line_endings("a\nb\rc"))
emit("bom1", strip_bom("\ufeffabc\n"))  # note: repr of bom char
emit("bom2", strip_bom("abc\n"))
emit("bom1b", [c for c in strip_bom("\ufeffabc\n")[0]])
emit("diff1", repr(generate_diff_with_hashes("a\nb\nc\n", "a\nB\nc\nd\n")))
emit("diff2", repr(generate_diff_with_hashes("x\n", "x\nsolo\n")))
emit("udiff1", repr(generate_diff_string("alpha\nbeta\n", "alpha\nBETA\n", 1)))

print("\n".join(out))
