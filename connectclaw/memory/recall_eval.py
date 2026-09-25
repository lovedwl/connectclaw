"""召回决策评估集落盘。

把每次「召回 → 使用确认」生命周期记成一条可回放记录，攒成离线评估集：
后续调阈值、上判别模型（Laya / AgentJev / 云端 Jev），效果好坏都拿它对比。
没有它，任何召回改动的收益只能靠体感。

一条记录 = 一次 recall() 事件 + 对应 confirm_usage() 给出的逐条标签：

    {"v", "ts", "session_id", "conversation_key", "query",
     "candidates": [{id, type, score, similarity, bm25, detail,
                     newly_injected, strength, importance, access_count, age_days}],
     "used_ids": [...], "reply"}

约定：
- persona 常驻条目（score==1.0，每轮必进、确认必中）不是召回决策，不记录——
  它会把注入精确率虚标到 100%，留着的只有噪音。
- ``newly_injected`` 标记本轮真的通过增量 op 进入上下文的条目；上一轮已在
  上下文里的召回结果不算本轮的决策，评估注入精确率时应排除。
- fail-open：任何异常只 debug，绝不影响召回主链路。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from connectclaw.logging import get_logger

from .types import SearchResult

logger = get_logger(__name__)

_RECORD_VERSION = 1
_QUERY_MAX_CHARS = 400
_REPLY_MAX_CHARS = 400
# confirm_usage 迟迟不来时的积压上限（多会话交错 / 异常轮次），丢最旧的。
_MAX_PENDING = 64


def _candidate_row(r: SearchResult, *, newly_injected: bool) -> dict:
    e = r.entry
    created = e.created_at or 0.0
    return {
        "id": e.id,
        "type": getattr(e.type, "value", str(e.type)),
        "score": round(float(r.score), 4),
        "similarity": round(float(r.similarity), 4),
        "bm25": round(float(r.bm25), 4),
        "detail": r.detail_level,
        "newly_injected": newly_injected,
        "strength": e.strength,
        "importance": e.importance,
        "access_count": e.access_count,
        "age_days": round(max(0.0, time.time() - created) / 86400.0, 3) if created else None,
    }


class RecallEvalLogger:
    """把召回决策写成 JSONL 评估集。所有方法 fail-open，绝不抛异常。"""

    def __init__(self, path: str | Path):
        self._path = Path(path).expanduser()
        # key 是 id(recalled_results 列表)；同时持有列表引用，GC 不会回收，
        # id 也就不会被复用——这是「同一批结果回传」最省事的配对方式。
        self._pending: dict[int, tuple[list, dict]] = {}

    def start(
        self,
        *,
        query: str,
        session_id: str,
        conversation_key: str,
        recalled_results: list,
        injected_ids: set[str] | None = None,
    ) -> None:
        """recall() 结束时调用：记下候选与注入决策，等 confirm_usage 收尾。"""
        try:
            if not recalled_results:
                return
            injected = injected_ids or set()
            candidates = [
                _candidate_row(r, newly_injected=r.entry.id in injected)
                for r in recalled_results
                if r.score != 1.0  # persona 常驻，不是决策
            ]
            if not candidates:
                return
            record = {
                "v": _RECORD_VERSION,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "session_id": session_id,
                "conversation_key": conversation_key,
                "query": (query or "")[:_QUERY_MAX_CHARS],
                "candidates": candidates,
            }
            self._pending[id(recalled_results)] = (recalled_results, record)
            while len(self._pending) > _MAX_PENDING:
                del self._pending[next(iter(self._pending))]
        except Exception as e:  # fail-open
            logger.debug("RecallEval: start failed: %s", e)

    def finalize(self, recalled_results: list, reply_text: str, used_ids: list[str]) -> None:
        """confirm_usage 后调用：补上逐条使用标签并落盘。未配对的 start 直接丢弃。"""
        try:
            pending = self._pending.pop(id(recalled_results), None)
            if pending is None:
                return
            _, record = pending
            record["used_ids"] = list(used_ids)
            record["reply"] = (reply_text or "")[:_REPLY_MAX_CHARS]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:  # fail-open
            logger.debug("RecallEval: finalize failed: %s", e)
