#!/usr/bin/env python3
"""召回决策评估：当前启发式的注入质量 vs confirm_usage 标签。

数据源是 ~/.connectclaw/memory_recall_eval.jsonl（MemorySubsystem 自动落盘，
每条记录 = 一次召回事件 + 逐条"被确认使用"标签）。没有评估集，调阈值或上
判别模型（Laya / AgentJev / Jev）都只能靠体感——本脚本就是那个"对比基准"。

指标约定：
- 精确率（precision）：本轮新注入的记忆里，多少被回复确认用上。越低说明
  塞了越多没用的（"多余"问题的直接度量）。
- 覆盖率（coverage）：被确认用上的记忆里，多少在本轮召回候选中。注意这只是
  "候选内覆盖"——用上了但根本没被召回的，日志里观测不到（没有负样本流）。
- 分数分离度：used vs not-used 的 score / similarity 均值与中位数。分离清晰
  → 调阈值就够；严重重叠 → 才需要判别模型。
- 相对下限回放：对记录里的 similarity 重放不同 keep_ratio，看精确率/覆盖率的
  权衡曲线，直接服务于当前启发式的调参。

用法：
    .venv/bin/python scripts/eval_memory_recall.py
    .venv/bin/python scripts/eval_memory_recall.py --path /tmp/xx.jsonl --last 500
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

DEFAULT_PATH = Path("~/.connectclaw/memory_recall_eval.jsonl").expanduser()


def load_records(path: Path, last: int | None) -> list[dict]:
    if not path.exists():
        print(f"评估集不存在：{path}")
        print("先让 bot 跑一段时间，MemorySubsystem 会自动落盘（每轮一条）。")
        sys.exit(1)
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("v") == 1 and rec.get("candidates"):
                records.append(rec)
    if last:
        records = records[-last:]
    return records


def mean_or_dash(xs: list[float]) -> str:
    return f"{statistics.mean(xs):.3f}" if xs else "-"


def median_or_dash(xs: list[float]) -> str:
    return f"{statistics.median(xs):.3f}" if xs else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--last", type=int, default=None, help="只评估最近 N 条记录")
    ap.add_argument(
        "--keep-ratios", type=float, nargs="*",
        default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        help="相对相关性下限回放的 keep_ratio 网格",
    )
    args = ap.parse_args()

    records = load_records(args.path, args.last)
    if not records:
        print("评估集为空（或全是旧版本记录）。")
        return

    # 展平候选；persona 已在落盘时排除
    cands: list[dict] = []
    for rec in records:
        used = set(rec.get("used_ids") or [])
        for c in rec["candidates"]:
            c = dict(c)
            c["used"] = c["id"] in used
            c["turn_injected"] = c.get("newly_injected", False)
            cands.append(c)

    injected = [c for c in cands if c["turn_injected"]]
    used_all = [c for c in cands if c["used"]]

    print(f"记录 {len(records)} 轮，候选 {len(cands)} 条，新注入 {len(injected)} 条，"
          f"确认使用 {len(used_all)} 条\n")

    # ── 1. 注入精确率 / 候选内覆盖 ──
    if injected:
        hits = sum(1 for c in injected if c["used"])
        print(f"注入精确率：{hits}/{len(injected)} = {hits / len(injected):.1%}"
              f"  （新塞进上下文的记忆里，回复真的用上的比例）")
    if used_all:
        cov = sum(1 for c in used_all if c["turn_injected"])
        print(f"候选内覆盖率：{cov}/{len(used_all)} = {cov / len(used_all):.1%}"
              f"  （用上的记忆里，本轮候选含它的比例）")
        print("  注：'用上了但没被召回'的样本日志里观测不到，这是该指标的边界。\n")

    per_type: dict[str, list[dict]] = {}
    for c in injected:
        per_type.setdefault(c.get("type", "?"), []).append(c)
    if per_type:
        print("分类型注入精确率：")
        for t, cs in sorted(per_type.items()):
            hits = sum(1 for c in cs if c["used"])
            print(f"  {t:<12} {hits}/{len(cs)} = {hits / len(cs):.1%}")
        print()

    # ── 2. 分数分离度 ──
    print("分数分离度（used vs not-used，非 persona 候选）：")
    for field in ("score", "similarity"):
        u = [c[field] for c in cands if c["used"] and c[field]]
        nu = [c[field] for c in cands if not c["used"] and c[field]]
        print(f"  {field:<11} used: 均值 {mean_or_dash(u)} / 中位 {median_or_dash(u)}"
              f"   not-used: 均值 {mean_or_dash(nu)} / 中位 {median_or_dash(nu)}")
    print("  分离清晰 → 调阈值即可；重叠严重 → 值得上判别模型。\n")

    # ── 3. 相对下限回放 ──
    # 仅用带 embedding 相似度的候选（keyword-only 召回的 similarity=0，没有可比性）
    sim_records = [
        rec for rec in records
        if any(c.get("similarity", 0) > 0 for c in rec["candidates"])
    ]
    if sim_records and args.keep_ratios:
        print(f"相对下限回放（{len(sim_records)} 轮有相似度信号）：")
        print("  ratio  保留率   注入精确率  候选内覆盖")
        for ratio in args.keep_ratios:
            kept_all, kept_used = 0, 0
            for rec in sim_records:
                sims = [c["similarity"] for c in rec["candidates"] if c["similarity"] > 0]
                if not sims:
                    continue
                cut = max(sims) * ratio
                kept = [c for c in rec["candidates"]
                        if c["similarity"] >= cut or c.get("bm25", 0) > 0]
                kept_all += len(kept)
                kept_used += sum(1 for c in kept if c["used"])
            prec = kept_used / kept_all if kept_all else 0.0
            total_used = sum(
                1 for rec in sim_records for c in rec["candidates"] if c["used"]
            )
            cov = kept_used / total_used if total_used else 0.0
            print(f"  {ratio:<6} {kept_all / max(1, len(sim_records)):<8.2f} "
                  f"{prec:<11.1%} {cov:.1%}")
        print("  保留率 = 每轮平均保留条数（相对召回总量）；"
              "精确率与覆盖率的平衡点就是当前 keep_ratio 该在的位置。")


if __name__ == "__main__":
    main()
