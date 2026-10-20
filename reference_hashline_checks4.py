"""Part 4: real recovery-success, three_way_merge direct, diff edge cases."""
import json
import sys
sys.path.insert(0, ".")

from connectclaw.hashline.hash import compute_line_hash
from connectclaw.hashline.parse import resolve_edit_anchors
from connectclaw.hashline.apply import execute_edit_pipeline
from connectclaw.hashline.snapshot import remember_read_snapshot, reset_read_snapshots
from connectclaw.hashline.merge import three_way_merge
from connectclaw.hashline.diff_util import generate_diff_with_hashes, generate_diff_string, detect_line_ending

out = []
emit = lambda *a: out.append(" ".join(repr(x) for x in a))

def anchor_for_line(content, lineno):
    lines = content.split("\n")
    if content.endswith("\n"):
        lines = lines[:-1]
    return f"{lineno}#{compute_line_hash(lines, lineno - 1)}"

# REAL recovery: edit at line1 (index1); far change at line5 in live
base = "l0\nl1\nl2\nl3\nl4\nl5\nl6\nl7\n"
live = "l0\nl1\nl2\nl3\nl4\nX5\nl6\nl7\n"
reset_read_snapshots()
remember_read_snapshot("/recover.py", base)
edits = resolve_edit_anchors([{"op": "replace", "pos": anchor_for_line(base, 2), "lines": ["NEW"]}])
try:
    r, rec = execute_edit_pipeline("/recover.py", edits, live)
    emit("recover_success", rec, repr(r.content), r.first_changed_line, r.last_changed_line, r.warnings)
except ValueError as e:
    emit("recover_success_ERR", str(e))

# three_way_merge direct
base_edited = "l0\nNEW\nl2\nl3\nl4\nl5\nl6\nl7\n"
emit("twm1", repr(three_way_merge(base, base_edited, live)))
emit("twm2", repr(three_way_merge(base, base, "x")))
emit("twm3", repr(three_way_merge(base, base_edited, base_edited)))
emit("twm4", repr(three_way_merge(base, base_edited, "l0\nl1\nl2\nl3\nl4\nl5\nl6\nl7\nextra\n")))
emit("twm5", repr(three_way_merge("a\nb\n", "a\nB\n", "a\nb\nc\n")))
emit("twm6", repr(three_way_merge("", "", "")))
emit("twm7", repr(three_way_merge("x", "y", "x")))
emit("twm8", repr(three_way_merge("x\ny\n", "x\ny\nz\n", "x\ny\nz\ny\n")))

# empty merge cases
emit("twm9", repr(three_way_merge("a\n", "b\n", "a\n")))

# diff on misc
emit("diff3", repr(generate_diff_with_hashes("x\ny\n", "y\nz\n")))
emit("diff4", repr(generate_diff_with_hashes("a\r\nb\r\n", "a\r\nb\r\nc\r\n")))
emit("dle_mixed", detect_line_ending("a\r\nb\nc\n"))

print("\n".join(out))
