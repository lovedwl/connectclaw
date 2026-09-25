"""一次性迁移：把旧格式会话里的"注入文本"转成结构化的 context 条目。

背景
----
2026-09-25 之前，一轮的注入上下文是**渲染成文本、拼进用户消息内容**里的；
之后改成独立的 `context` 条目（结构化 ops，见 connectclaw/injection.py）。
本脚本把历史会话就地迁移一次，于是运行时代码里那条"解析旧文本"的兼容路径
可以彻底删掉（一个机制，而不是两个）。

迁移做的事（每个会话文件）
--------------------------
1. ``message/user`` 且内容里含注入块 → 拆成
   ``context`` 条目（ops = memory_add / catalog_set）+ 干净的用户消息；父链改接；
2. ``compaction`` 有 ``merged_context`` 但没有 ``context_state`` → 解析一次补上结构化快照。

用法
----
    python scripts/migrate_sessions_to_structured.py --dry-run    # 只报告
    python scripts/migrate_sessions_to_structured.py --apply      # 就地改写（逐文件备份为 .bak-<时间戳>）
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
import uuid

# ── 旧格式的解析规则（只存在于这个一次性脚本里，运行时不带）──
_SECTION_LINES = r"(?:(?!\n)[^\n]+(?:\n|\Z))*"
_AGENTS_HEAD = r"##\s*可运行的 agents[^\n]*(?:\n|\Z)"
_TOOLS_HEAD = r"##\s*可授权给子 agent 的工具[^\n]*(?:\n|\Z)"

_PATTERNS = (
    re.compile(r"\A\s*<remembered-context>.*?</remembered-context>", re.S),
    re.compile(r"\A\s*<retrieved-documents>.*?</retrieved-documents>", re.S),
    re.compile(r"\A\s*" + _AGENTS_HEAD + _SECTION_LINES),
    re.compile(r"\A\s*" + _TOOLS_HEAD + _SECTION_LINES),
)
_MEMORY_BLOCK_SEARCH = re.compile(r"<remembered-context>(.*?)</remembered-context>", re.S)
_CATALOG_SEARCH = re.compile(
    _AGENTS_HEAD + _SECTION_LINES + r"(?:\n*" + _TOOLS_HEAD + _SECTION_LINES + r")?",
    re.M,
)
_FORGOTTEN_PREFIX = "已遗忘："


def split_injected(text: str) -> tuple[str, str]:
    """旧格式：把开头的注入段与用户正文分开（返回 ``(注入段, 正文)``）。"""
    if not text:
        return "", ""
    injected: list[str] = []
    rest = text
    while True:
        for pattern in _PATTERNS:
            match = pattern.match(rest)
            if match:
                injected.append(match.group(0).strip())
                rest = rest[match.end():]
                break
        else:
            break
    if not injected:
        return "", text
    return "\n\n".join(injected), rest.lstrip("\n \t")


def _memory_lines(text: str) -> list[str]:
    match = _MEMORY_BLOCK_SEARCH.search(text or "")
    if not match:
        return []
    lines: list[str] = []
    current: str | None = None
    for raw in match.group(1).split("\n"):
        if raw.startswith("- "):
            if current:
                lines.append(current)
            current = raw
        elif current is not None and raw.startswith("  "):
            current += "\n" + raw
    if current:
        lines.append(current)
    return lines


def _line_content(line: str) -> str:
    text = re.sub(r"^-\s+", "", line or "")
    text = re.sub(r"^\[[^\]]*\]\s*", "", text)
    text = re.sub(r"^\[(?:past|pattern)\]\s*", "", text)
    return text.split("\n")[0].strip()


def ops_from_text(text: str) -> list[dict]:
    """旧文本 → ops（合成 id 带 legacy: 前缀，便于人看出来源）。"""
    ops: list[dict] = []
    for line in _memory_lines(text):
        content = _line_content(line)
        if not content:
            continue
        if content.startswith(_FORGOTTEN_PREFIX):
            target = " ".join(content[len(_FORGOTTEN_PREFIX):].split())
            ops.append({"op": "memory_forget", "id": f"legacy:{target}"})
            continue
        ops.append({"op": "memory_add", "id": f"legacy:{content}", "line": line, "content": content})
    match = _CATALOG_SEARCH.search(text or "")
    if match:
        ops.append({"op": "catalog_set", "text": match.group(0).strip()})
    return ops


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _with_clean_text(content: object, clean: str) -> object:
    """把消息内容里的注入剥掉，保留其它块（图片等）。"""
    if isinstance(content, str):
        return clean
    if isinstance(content, list):
        out: list[dict] = []
        first = True
        for block in content:
            if (first and isinstance(block, dict) and block.get("type") == "text"
                    and "<remembered-context>" in str(block.get("text", ""))):
                out.append({**block, "text": clean})
                first = False
            else:
                out.append(block)
        return out
    return content


def new_id(existing: set[str]) -> str:
    while True:
        eid = str(uuid.uuid4()).replace("-", "")[:12]
        if eid not in existing:
            return eid


def migrate_file(path: str, existing: set[str]) -> tuple[list[dict], dict]:
    """返回 (新行, 统计)。"""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    stats = {"contexts_added": 0, "compactions_annotated": 0, "only_injection": 0}
    out: list[dict] = []

    for row in rows:
        etype = row.get("type")

        if etype == "message":
            msg = row.get("message") or {}
            if msg.get("role") == "user":
                text = _text_of(msg.get("content"))
                injected, clean = split_injected(text)
                if injected:
                    ctx_id = new_id(existing)
                    existing.add(ctx_id)
                    out.append({
                        "type": "context",
                        "id": ctx_id,
                        "parent_id": row.get("parent_id"),
                        "timestamp": row.get("timestamp", ""),
                        "ops": ops_from_text(injected),
                        "note": "迁移",
                    })
                    stats["contexts_added"] += 1
                    if not clean.strip():
                        stats["only_injection"] += 1
                    row = dict(row)
                    row["parent_id"] = ctx_id
                    row["message"] = {**msg, "content": _with_clean_text(msg.get("content"), clean)}
            out.append(row)
            continue

        if etype == "compaction" and row.get("merged_context") and not row.get("context_state"):
            ops = ops_from_text(row["merged_context"])
            row = dict(row)
            row["context_state"] = {
                "op": "state_snapshot",
                "memory": {o["id"]: {"line": o["line"], "content": o["content"]}
                           for o in ops if o.get("op") == "memory_add"},
                "catalog": next((o["text"] for o in ops if o.get("op") == "catalog_set"), ""),
            }
            stats["compactions_annotated"] += 1

        out.append(row)

    return out, stats


def validate(path: str, rows: list[dict]) -> list[str]:
    """结构自检：id 唯一、父链可达、消息不含注入。"""
    problems: list[str] = []
    ids: set[str] = set()
    parents: dict[str, str | None] = {}
    for row in rows:
        eid = row.get("id")
        if not eid:
            problems.append("有条目缺 id")
            continue
        if eid in ids:
            problems.append(f"id 重复: {eid}")
        ids.add(eid)
        parents[eid] = row.get("parent_id")

    for eid, parent in parents.items():
        if parent and parent not in ids:
            problems.append(f"父链断裂: {eid} → {parent}")
    for row in rows:
        if row.get("type") != "message":
            continue
        msg = row.get("message") or {}
        # 只查**用户消息**：toolResult 里出现这段文本是正常的（工具读过会话文件、
        # 把里面的历史注入当内容返回了），助手引用它也是正常的。
        if msg.get("role") != "user":
            continue
        text = _text_of(msg.get("content"))
        if "<remembered-context>" in text:
            problems.append(f"用户消息仍含注入文本: {row.get('id')}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", default="~/.connectclaw/sessions")
    ap.add_argument("--apply", action="store_true", help="就地改写（默认只报告）")
    ap.add_argument("--dry-run", action="store_true", help="只报告（默认行为）")
    args = ap.parse_args()

    sdir = os.path.expanduser(args.sessions_dir)
    files = sorted(glob.glob(os.path.join(sdir, "*.jsonl")))
    if not files:
        print("没有会话文件:", sdir)
        return 1

    stamp = time.strftime("%Y%m%d-%H%M%S")
    total = {"contexts_added": 0, "compactions_annotated": 0, "only_injection": 0}
    print(f"{'文件':<26} {'新增context':>10} {'压缩补快照':>10} {'整条全注入':>10}")
    for path in files:
        try:
            rows, stats = migrate_file(path, set())
        except Exception as e:  # noqa: BLE001
            print(f"  {os.path.basename(path):<24} 解析失败: {type(e).__name__}: {e}")
            continue
        for key in total:
            total[key] += stats[key]
        changed = stats["contexts_added"] or stats["compactions_annotated"]
        print(f"  {os.path.basename(path):<24} {stats['contexts_added']:>10} "
              f"{stats['compactions_annotated']:>10} {stats['only_injection']:>10}"
              + ("  → 会改写" if changed else "  （无需改动）"))

        if args.apply and changed:
            problems = validate(path, rows)
            if problems:
                print(f"    ✗ 自检未过，跳过改写: {problems[:3]}")
                continue
            shutil.copy2(path, f"{path}.bak-{stamp}")
            if os.path.getsize(path) > 3_000_000:
                print("    ← 注意: 这是个大会话文件，改写前请确认备份完好")
            with open(path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
            # 复读校验
            with open(path, encoding="utf-8") as fh:
                reread = [json.loads(l) for l in fh if l.strip()]
            problems = validate(path, reread)
            print("    ✓ 已改写并通过自检" if not problems else f"    ✗ 改写后自检未过: {problems[:3]}")

    print(f"\n合计：新增 context 条目 {total['contexts_added']}、压缩补快照 "
          f"{total['compactions_annotated']}、整条只有注入的 {total['only_injection']}")
    if not args.apply:
        print("（这是干跑；确认无误后加 --apply 就地改写，逐文件备份 .bak-<时间戳>）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
