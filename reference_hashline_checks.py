"""Generate reference outputs for hashline Rust port verification."""
import json
import sys

sys.path.insert(0, ".")

from connectclaw.hashline.hash import (
    compute_hash_from_context, compute_line_hash, normalize_hash_input,
    normalize_fuzzy_line, is_fuzzy_equivalent_line, NIBBLE_STR,
)
from connectclaw.hashline.parse import (
    parse_anchor_ref, resolve_edit_anchors, normalize_edit_request, Anchor,
)
from connectclaw.hashline.apply import (
    apply_hashline_edits, execute_edit_pipeline, ApplyResult, NoopEdit,
)
from connectclaw.hashline.format import (
    compute_affected_line_range, compute_changed_line_range,
    format_hashline_region,
)
from connectclaw.hashline.snapshot import (
    remember_read_snapshot, get_read_snapshot, get_read_snapshot_versions,
    reset_read_snapshots,
)
from connectclaw.hashline.guard import (
    record_noop_edit, record_applied_edit, is_duplicate_applied_payload,
    clear_applied_payload, reset_noop_guard,
)

out = []
emit = lambda *a: out.append(" ".join(repr(x) for x in a))

# ── hash section ──────────────────────────────────────────────
pairs = [
    ("a", "b", "c"),
    ("", "a", ""),
    ("", "", ""),
    ("x", "y", "z"),
    ("foo", "bar", "baz"),
    ("fn main() {", '    println!("hi");', "}"),
    ("line with \ttab", "  leading spaces", "trailing   \t "),
    ("中文内容", "第二行", "第三行"),
    ("", "single", ""),
    ("prev", "", "next"),
]
for p, c, n in pairs:
    emit("hc", p, c, n, "=>", compute_hash_from_context(p, c, n))

lines = ["a", "b", "c"]
for i in range(len(lines)):
    emit("clh", i, "=>", compute_line_hash(lines, i))

lines2 = ['fn main() {', '    println!("hi");', '}']
for i in range(len(lines2)):
    emit("clh2", i, "=>", compute_line_hash(lines2, i))

# normalize
emit("norm1", repr(normalize_hash_input("  a  \r\n")))
emit("norm2", repr(normalize_hash_input("x")))
emit("norm3", repr(normalize_hash_input("")))

# fuzzy
emit("fuzzy1", repr(normalize_fuzzy_line("Hello\u2018World\u2019\u2014x\u00a0y")))
emit("fuzzy2", repr(normalize_fuzzy_line("a \u201cquoted\u201d \u2013 line")))
emit("fuzzy_eq1", is_fuzzy_equivalent_line("a\u2018b", "a'b"))
emit("fuzzy_eq2", is_fuzzy_equivalent_line("x\u2014y", "x - y"))
emit("fuzzy_eq3", is_fuzzy_equivalent_line("  pad  ", "pad"))
emit("fuzzy_eq4", is_fuzzy_equivalent_line("same", "different"))
emit("nibble", NIBBLE_STR)

# ── parse section ─────────────────────────────────────────────
def try_parse(s):
    try:
        a = parse_anchor_ref(s)
        return f"OK({a.line},{a.hash!r},hint={a.text_hint!r})"
    except ValueError as e:
        return f"ERR: {e}"

for s in [
    "5#MQQV",
    "5#MQQV:some content",
    "  > 5#MQQV  ",
    "  5 # KY:text",
    "1#AB",
    "0#AB",
    "5#ZZ",
    "5#ABC",
    "5#1234",
    "5#AB_",
    "5",
    "5:",
    "5#",
    "5##",
    "",
    "   ",
    "abc#DE",
    "123#:",
    "1#A",
    "x",
]:
    emit("parse", s, "=>", try_parse(s))

# resolve_edit_anchors
edits = [
    {"op": "replace", "pos": "1#KY", "lines": ["a", "b"]},
    {"op": "append", "pos": "2#HY", "lines": ["zz"]},
    {"op": "prepend", "lines": ["head"]},
    {"op": "replace_text", "oldText": "o", "newText": "n"},
    {"op": "replace", "pos": "1#KY", "end": "2#HY", "lines": ["c"]},
]
try:
    res = resolve_edit_anchors(edits)
    for r in res:
        if hasattr(r, "op"):
            fields = {k: v for k, v in r.__dict__.items() if k != "op"}
            emit("resolved", r.op, "=>", fields)
except ValueError as e:
    emit("resolve_ERR", str(e))

# bad op
for bad in [
    [{"op": "bogus", "lines": ["x"]}],
    [{"op": "replace", "lines": ["x"]}],
    [{"op": "replace", "pos": "1#KY", "lines": ["x"], "extra": 1}],
    [{"op": "append", "lines": ["x"], "end": "1#KY"}],
    [{"op": "replace_text", "oldText": 3, "newText": "x"}],
    [{"op": "append", "lines": ["3#ZZ\nnot-pre"]}],
    [{"op": "replace", "pos": "1#KY", "lines": ["x"], "oldText": "a", "newText": "b"}],
]:
    try:
        r = resolve_edit_anchors(bad)
        emit("resolve_bad_ok", json.dumps(bad, ensure_ascii=False), "=>", len(r))
    except ValueError as e:
        emit("resolve_bad_ERR", json.dumps(bad, ensure_ascii=False), "=>", e)

# display prefix rejection inside resolve (lines containing hashline prefixes)
prefix_cases = [
    [{"op": "append", "lines": [">>> 3#ZZ:content"]}],
    [{"op": "append", "lines": ["3#ZZ:content"]}],
    [{"op": "prepend", "lines": ["+ 3#ZZ:content"]}],
    [{"op": "prepend", "lines": ["- 3     old line"]}],
    [{"op": "replace", "pos": "1#KY", "lines": ["#ZZ:x"]}],
    [{"op": "replace", "pos": "1#KY", "lines": [">>2#AB:x"]}],
]
for case in prefix_cases:
    try:
        r = resolve_edit_anchors(case)
        emit("prefix_ok", json.dumps(case, ensure_ascii=False), "=>", len(r))
    except ValueError as e:
        emit("prefix_ERR", json.dumps(case, ensure_ascii=False), "=>", e)

# normalize_edit_request
norm_cases = [
    {"path": "/x.py", "edits": [{"op": "append", "lines": ["a"]}]},
    {"file_path": "/x.py", "edits": "[]"},
    {"file_path": "/x.py", "oldText": "a", "newText": "b"},
    {"path": "/x.py", "old_text": "a", "new_text": "b"},
    {"path": "/x.py", "edits": []},
    {"path": "/x.py", "edits": '[{"oldText": "a", "newText": "b"}]'},
    {"path": "/x.py", "edits": [{"oldText": "a", "newText": "b"}]},
    {"path": "/x.py", "edits": [{"op": "replace", "pos": "1#KY", "lines": ["z"]}], "oldText": "a", "newText": "b"},
    "not-a-dict",
    {"path": "/x.py", "edits": "not json at all"},
    {"path": "/x.py", "oldText": "a"},
    {"path": "/x.py", "oldText": 5, "newText": "b"},
    {"path": "/x.py", "oldText": "a", "old_text": "b", "newText": "c", "new_text": "d"},
]
for c in norm_cases:
    try:
        emit("norm_req", json.dumps(c, ensure_ascii=False), "=>", json.dumps(normalize_edit_request(c), ensure_ascii=False))
    except ValueError as e:
        emit("norm_req_ERR", json.dumps(c, ensure_ascii=False), "=>", e)

# ── apply section ─────────────────────────────────────────────
CONTENT = "alpha\nbeta\ngamma\ndelta\n"

def run_apply(content, edits, label):
    try:
        resolved = resolve_edit_anchors(edits)
        r = apply_hashline_edits(content, resolved)
        noops = [(n.edit_index, n.loc, n.current_content) for n in r.noop_edits]
        emit("apply", label,
             "content=", repr(r.content),
             "first=", r.first_changed_line, "last=", r.last_changed_line,
             "warnings=", r.warnings, "noops=", noops)
    except ValueError as e:
        emit("apply_ERR", label, "=>", e)

def anchor_for_line(content, lineno):
    lines = content.split("\n")
    if content.endswith("\n"):
        lines = lines[:-1]
    return f"{lineno}#{compute_line_hash(lines, lineno - 1)}"

# determine hashes for content lines
cl = CONTENT.split("\n")[:-1]
emit("content_hashes", [compute_line_hash(cl, i) for i in range(len(cl))])

# replace single line
run_apply(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 2), "lines": ["BETA"]}], "replace_single")
# replace multi-line content
run_apply(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 1), "end": anchor_for_line(CONTENT, 2), "lines": ["A", "B", "C"]}], "replace_range")
# append at pos
run_apply(CONTENT, [{"op": "append", "pos": anchor_for_line(CONTENT, 2), "lines": ["new2", "new3"]}], "append_pos")
# append at EOF (no pos) — content has trailing newline
run_apply(CONTENT, [{"op": "append", "lines": ["tail"]}], "append_eof_trailing")
# append at EOF (no pos) — content has no trailing newline
run_apply("alpha\nbeta", [{"op": "append", "lines": ["tail"]}], "append_eof_no_trailing")
# append sentinel: pos on last visible line when terminal newline
run_apply(CONTENT, [{"op": "append", "pos": anchor_for_line(CONTENT, 4), "lines": ["tail"]}], "append_sentinel")
# prepend at pos
run_apply(CONTENT, [{"op": "prepend", "pos": anchor_for_line(CONTENT, 2), "lines": ["p1"]}], "prepend_pos")
# prepend at start (no pos)
run_apply(CONTENT, [{"op": "prepend", "lines": ["header"]}], "prepend_start")
# replace_text
run_apply(CONTENT, [{"op": "replace_text", "oldText": "gamma", "newText": "GAMMA"}], "replace_text_1")
# empty file append
run_apply("", [{"op": "append", "lines": ["a", "b"]}], "append_empty")
# empty file prepend
run_apply("", [{"op": "prepend", "lines": ["a"]}], "prepend_empty")
# noop replace (same content)
run_apply(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 2), "lines": ["beta"]}], "replace_noop")
# noop replace_text same
run_apply(CONTENT, [{"op": "replace_text", "oldText": "gamma", "newText": "gamma"}], "replace_text_noop")
# replace_text no match
run_apply(CONTENT, [{"op": "replace_text", "oldText": "nonexistent", "newText": "x"}], "replace_text_nomatch")
# replace_text multiple match
run_apply("aa\naa\n", [{"op": "replace_text", "oldText": "aa", "newText": "bb"}], "replace_text_multimatch")
# overlapping replace_text matches
run_apply("aaa", [{"op": "replace_text", "oldText": "aa", "newText": "x"}], "replace_text_overlap")
# stale anchor
run_apply("alpha\nBETA\ngamma\ndelta\n", [{"op": "replace", "pos": anchor_for_line("alpha\nbeta\ngamma\ndelta\n", 2), "lines": ["x"]}], "stale_anchor")
# fuzzy anchor accepted (hash mismatched but hinted content matches)
run_apply(CONTENT, [{"op": "replace", "pos": "2#WRONG0:beta", "lines": ["x"]}], "fuzzy_wrong_hash")
# fuzzy anchor with correct hint content but wrong hash
# give a real wrong-hash-but-hint case:
real_hash2 = compute_line_hash(CONTENT.split("\n")[:-1], 1)
fake = ("Q" if real_hash2[0] != "Q" else "W") + real_hash2[1:]
run_apply(CONTENT, [{"op": "replace", "pos": f"2#{fake}:beta", "lines": ["x"]}], "fuzzy_hint_forgave")
# text_hint mismatch anti-collision: correct hash but wrong hint
run_apply(CONTENT, [{"op": "replace", "pos": f"2#{real_hash2}:WRONGHINT", "lines": ["x"]}], "anti_collision")
# empty lines replace
run_apply(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 2), "end": anchor_for_line(CONTENT, 3), "lines": []}], "replace_to_empty_middle")
# replace to empty at end
run_apply(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 4), "lines": []}], "replace_last_to_empty")
# replace to empty whole file
run_apply(CONTENT, [{"op": "replace", "pos": "1#" + compute_line_hash(CONTENT.split("\n")[:-1], 0), "end": "4#" + compute_line_hash(CONTENT.split("\n")[:-1], 3), "lines": []}], "replace_whole_to_empty")
# conflicting inserts
run_apply(CONTENT, [{"op": "append", "pos": anchor_for_line(CONTENT, 2), "lines": ["x"]}, {"op": "append", "pos": anchor_for_line(CONTENT, 2), "lines": ["y"]}], "conflict_inserts")
# overlapping replaces
run_apply(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 1), "end": anchor_for_line(CONTENT, 3), "lines": ["x"]}, {"op": "replace", "pos": anchor_for_line(CONTENT, 2), "end": anchor_for_line(CONTENT, 2), "lines": ["y"]}], "overlap_replace")
# out of bounds
run_apply(CONTENT, [{"op": "replace", "pos": "99#AB", "lines": ["x"]}], "oob")
# empty edits -> identity
emit("apply_empty_edits", repr(apply_hashline_edits(CONTENT, []).content))
# does-not-empty guard: replace_text to empty on nonempty... can't easily empty
# replace lines empty on single line of one-line file
run_apply("only", [{"op": "replace", "pos": "1#" + compute_line_hash(["only"], 0), "lines": []}], "empty_single_line_file_mid_no")
# dedup identical spans
run_apply(CONTENT, [{"op": "append", "pos": anchor_for_line(CONTENT, 2), "lines": ["dup"]}, {"op": "append", "pos": anchor_for_line(CONTENT, 2), "lines": ["dup"]}], "dedup_spans")

# range order bad
run_apply(CONTENT, [{"op": "replace", "pos": anchor_for_line(CONTENT, 3), "end": anchor_for_line(CONTENT, 1), "lines": ["x"]}], "bad_range_order")
# append sentinel without trailing newline: pos on last line
run_apply("alpha\nbeta", [{"op": "append", "pos": anchor_for_line("alpha\nbeta", 2), "lines": ["t"]}], "append_last_no_trailing")

print("\n".join(out))
