"""Part 3: pipeline success, fuzzy warning, candidates overflow, misc."""
import json
import sys

sys.path.insert(0, ".")

from connectclaw.hashline.hash import compute_line_hash, compute_hash_from_context
from connectclaw.hashline.parse import resolve_edit_anchors
from connectclaw.hashline.apply import apply_hashline_edits, execute_edit_pipeline
from connectclaw.hashline.snapshot import remember_read_snapshot, reset_read_snapshots

out = []
emit = lambda *a: out.append(" ".join(repr(x) for x in a))

def A(content, edits, label):
    try:
        r = apply_hashline_edits(content, resolve_edit_anchors(edits))
        return ("OK", repr(r.content), r.first_changed_line, r.last_changed_line, r.warnings)
    except ValueError as e:
        return ("ERR", str(e))

def anchor_for_line(content, lineno):
    lines = content.split("\n")
    if content.endswith("\n"):
        lines = lines[:-1]
    return f"{lineno}#{compute_line_hash(lines, lineno - 1)}"

# fuzzy acceptance (forgiveness) with smart quote difference
C_SQ = "it‘s\ntwo\nthree\nfour\n"
sq_lines = C_SQ.split("\n")[:-1]
ref_hash1 = compute_line_hash(sq_lines, 0)  # hash of raw line "it‘s"
# hint with ASCII quote -> different hash
hint_hash = compute_hash_from_context("", "it's", "two")
emit("sq_line_hash", ref_hash1, hint_hash, "eq?", ref_hash1 == hint_hash)
emit("forgive_smartquote", A(C_SQ, [{"op": "replace", "pos": f"1#{hint_hash}:it's", "lines": ["REPLACED"]}], "x"))

# candidate overflow: file with many "same" lines; old line1 anchor with hint "same"
EIGHT = "\n".join(["same"] * 10) + "\n"
# anchor: line 1 with a VALID-format but wrong hash; e.g. hash "AA"? A not in alphabet.
# choose a valid-format wrong hash not equal to actual line1 hash: actual hash of "same".
r1 = compute_line_hash(["same"] * 10, 0)
# pick a different valid 2-char
wrong = r1
alt = "WR" if wrong != "WR" else "WV"
emit("cand_overflow_basic", A(EIGHT, [{"op": "replace", "pos": f"1#{alt}:same", "lines": ["X"]}], "x"))

# candidate list (<=3 matches) far away unique lines
LINES = [f"u{i}" for i in range(30)]
C30 = "\n".join(LINES) + "\n"
ED_LINES = list(LINES); ED_LINES[2] = "CHANGED"
C30ED = "\n".join(ED_LINES) + "\n"
old_hash3 = compute_line_hash(LINES, 2)
emit("cand_list", A(C30ED, [{"op": "replace", "pos": f"3#{old_hash3}:u20", "lines": ["X"]}], "x"))

# multiple mismatches (range replace both ends stale)
C2 = "A1\nB2\nC3\nD4\nE5\n"
L2 = C2.split("\n")[:-1]
# anchor old hashes for line2, line4 from a DIFFERENT content
other = "a1\nb2\nc3\nd4\ne5\n"
OL = other.split("\n")[:-1]
oh2 = compute_line_hash(OL, 1)
oh4 = compute_line_hash(OL, 3)
emit("multi_mismatch", A(C2, [{"op": "replace", "pos": f"2#{oh2}", "end": f"4#{oh4}", "lines": ["X"]}], "x"))

# pipeline success recovery (trailing newline -> merged loses trailing newline?)
reset_read_snapshots()
path = "/tmp/succ.py"
old = "aaa\nbbb\nccc\n"
remember_read_snapshot(path, old)
edits = resolve_edit_anchors([{"op": "replace", "pos": anchor_for_line(old, 2), "lines": ["NEW"]}])
current = "aaa\nbbb\nccc\nddd\n"
try:
    r, rec = execute_edit_pipeline(path, edits, current)
    emit("pipeline_success", rec, repr(r.content), r.first_changed_line, r.last_changed_line, r.warnings)
except ValueError as e:
    emit("pipeline_success_ERR", str(e))

# pipeline with no snapshot -> stale anchor message unchanged (valid anchor incoming)
reset_read_snapshots()
current2 = "aaa\nBBB\nccc\nddd\n"
edits2 = resolve_edit_anchors([{"op": "replace", "pos": anchor_for_line("aaa\nbbb\nccc\n", 2), "lines": ["NEW"]}])
try:
    r, rec = execute_edit_pipeline("/nosnap.py", edits2, current2)
    emit("pipeline_nosnap_ok", rec, repr(r.content))
except ValueError as e:
    emit("pipeline_nosnap_ERR", str(e))

# reduce_context: many-line replace -> changed range wide; compute_affected_line_range None
emit("aff_wide", "skip")
# repair: replace_text with \r-only normalization
emit("rt_cr_only", A("a\rb\nc\n", [{"op": "replace_text", "oldText": "a\rb", "newText": "AB"}], "x"))
# rt separate-line: oldText spanning two lines with \n, content LF
emit("rt_multi", A("aaa\nbbb\nccc\n", [{"op": "replace_text", "oldText": "aaa\nbbb", "newText": "AB"}], "x"))

# normalize_edit_request with json edits + newline? and dict ordering (semantic check via parse)
from connectclaw.hashline.parse import normalize_edit_request
import connectclaw.hashline.parse as parse_mod
# multi-line edits-as-JSON-string
res = normalize_edit_request({"file_path": "/x", "edits": '[{"op":"replace","pos":"1#KY","lines":["a\\nb"]}]'})
emit("norm_json_multi", json.dumps(res, ensure_ascii=False))

print("\n".join(out))
