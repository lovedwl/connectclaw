"""上下文注入的结构化状态（单机制，无文本兼容路径）。

架构（2026-09-25 由用户拍板）：

    op（结构化、落盘）  --fold-->  ContextState（Python 对象）  --render-->  文本
                                                                        （只在组装上下文那一刻，
                                                                          且**冻结**进历史）

一轮的注入不再"渲染成文本拼进用户消息"，而是一条独立的 `context` 条目，内容是一组
op：`memory_add` / `memory_forget` / `catalog_set` / `state_snapshot`。

- **当前状态**由 `fold_ops` 从会话折叠出来（`AgentHarness.context_state()`），因此
  "哪些已经注入过"是**推导**出来的——没有内存账本，进程重启也不会整批重发。
- 增删改都是对象层面的操作，**合并/压缩**因此不再需要正则解析文本
  （压缩做的事就是 `state.snapshot_ops()` 拍一份快照）。
- op 里带着**当时渲染好的行文本**（`line`/`text`）：历史轮次的渲染是冻结的，这正是
  前缀缓存的要求（上游严格按请求前缀匹配；实测同一前缀第二次请求 cached_tokens=
  7168/7235，而一旦改写历史，命中就在改写点断开）。

**旧会话（注入还拼在消息文本里）已由 `scripts/migrate_sessions_to_structured.py`
一次性迁移**，所以这里不再保留任何"解析旧文本"的兼容代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from connectclaw.logging import get_logger

logger = get_logger(__name__)


# ── op 种类 ────────────────────────────────────────────────────

OP_MEMORY_ADD = "memory_add"
OP_MEMORY_FORGET = "memory_forget"
OP_CATALOG_SET = "catalog_set"
OP_STATE_SNAPSHOT = "state_snapshot"   # 压缩时把整份状态拍成一条（git squash 的 op 形态）


# ── 文本渲染（格式只在这里定义一份）────────────────────────────

INCREMENTAL_HEADER = (
    "(记忆更新：以下只列出本次新增或发生变化的条目，未列出的按更早轮次里的说法沿用。"
    "每条前缀 [日期 · 强度]：日期是记录时间，强度 0~1 是可信度——"
    "同主题说法冲突时以日期较新者为准，强度低者存疑)"
)
FULL_HEADER = (
    "(Things you know from past interactions. "
    "每条前缀 [日期 · 强度]：日期是记录时间，强度 0~1 是可信度——"
    "同主题说法冲突时以日期较新者为准，强度低者存疑)"
)
_MERGED_HEADER = (
    "(压缩后合并的记忆状态：以下是此前注入过的记忆的当前值。"
    "每条前缀 [日期 · 强度]：日期是记录时间，强度 0~1 是可信度)"
)


def render_memory_block(lines: list[str], *, header: str = "") -> str:
    """把若干条目行包成一个注入块。记忆检索、op 渲染都走它，别在别处再拼一遍。"""
    if not lines:
        return ""
    return "\n".join(
        ["<remembered-context>", header or FULL_HEADER, *lines, "</remembered-context>"]
    )


# ── 状态 ───────────────────────────────────────────────────────


@dataclass
class ContextState:
    """当前上下文状态（Python 对象，不是文本）。"""

    # memory_id -> {"line": 当时渲染好的那行, "content": 原始内容}
    memory: dict[str, dict[str, str]] = field(default_factory=dict)
    catalog: str = ""

    # ── 作用 op ────────────────────────────────────────────
    def apply(self, ops: list[dict[str, Any]]) -> None:
        for op in ops or []:
            kind = op.get("op")
            if kind == OP_MEMORY_ADD:
                mid = str(op.get("id") or "")
                if mid:
                    self.memory[mid] = {
                        "line": op.get("line", ""),
                        "content": op.get("content", ""),
                    }
            elif kind == OP_MEMORY_FORGET:
                self.memory.pop(str(op.get("id") or ""), None)
            elif kind == OP_CATALOG_SET:
                self.catalog = op.get("text", "") or ""
            elif kind == OP_STATE_SNAPSHOT:
                # 快照 = 整份替换。折叠时它会被放在保留区之前（见 harness.context_state），
                # 所以这里的"清空"语义是对的。
                self.memory = {
                    str(k): dict(v) for k, v in (op.get("memory") or {}).items()
                }
                self.catalog = op.get("catalog", "") or ""
            else:
                logger.debug("injection: 未知 op 类型 %r，忽略", kind)

    # ── 产出 op（增量计算都在对象层面）─────────────────────
    def memory_delta(
        self,
        items: list[tuple[str, str, str]],
        alive: Callable[[str], bool] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """``items = [(memory_id, 渲染文本, 原始内容)]`` → ``(add ops, forget ops)``。

        只按**内容**比对：渲染文本里带着强度值，而强度每轮都可能被 confirm_usage /
        衰减改动——拿它当键的话，每条都会每轮被判成"变了"，增量机制直接失效。
        """
        current = {mid: (line, content) for mid, line, content in items}

        add_ops: list[dict[str, Any]] = []
        for mid, (line, content) in current.items():
            prev = self.memory.get(mid)
            if prev is None or prev.get("content") != content:
                add_ops.append({
                    "op": OP_MEMORY_ADD, "id": mid, "line": line, "content": content,
                })

        forget_ops: list[dict[str, Any]] = []
        if alive is not None:
            for mid in list(self.memory):
                if mid in current:
                    continue
                if not alive(mid):
                    forget_ops.append({"op": OP_MEMORY_FORGET, "id": mid})
        return add_ops, forget_ops

    def catalog_delta(self, text: str) -> list[dict[str, Any]]:
        """清单变了才产出 op（首次也算变化：新会话要先交代一遍）。"""
        if not text or text == self.catalog:
            return []
        return [{"op": OP_CATALOG_SET, "text": text}]

    # ── 渲染（只在边界用）──────────────────────────────────
    def render(self, ops: list[dict[str, Any]], *, incremental: bool = True) -> str:
        """把一批 op 渲染成给模型的文本。

        增量 op → 只列新增行；快照 op → 全量（合并表头）；catalog 用 op 自带的文本。
        """
        snapshot = any(op.get("op") == OP_STATE_SNAPSHOT for op in ops or [])
        lines: list[str] = []
        catalogs: list[str] = []
        for op in ops or []:
            kind = op.get("op")
            if kind == OP_MEMORY_ADD:
                if op.get("line"):
                    lines.append(op["line"])
            elif kind == OP_CATALOG_SET:
                if op.get("text"):
                    catalogs.append(op["text"])
            elif kind == OP_STATE_SNAPSHOT:
                lines.extend(v.get("line", "") for v in (op.get("memory") or {}).values())
                if op.get("catalog"):
                    catalogs.append(op["catalog"])

        blocks: list[str] = []
        header = _MERGED_HEADER if snapshot else (INCREMENTAL_HEADER if incremental else FULL_HEADER)
        if lines:
            blocks.append(render_memory_block([ln for ln in lines if ln], header=header))
        blocks.extend(catalogs)
        return "\n\n".join(blocks)

    def snapshot_ops(self) -> list[dict[str, Any]]:
        """把当前状态拍成一条快照 op（压缩用）。"""
        return [{
            "op": OP_STATE_SNAPSHOT,
            "memory": {mid: dict(v) for mid, v in self.memory.items()},
            "catalog": self.catalog,
        }]


def render_ops(ops: list[dict[str, Any]], *, incremental: bool = True) -> str:
    """把一批 op 渲染成给模型的文本（纯函数，不做状态累积）。"""
    return ContextState().render(ops, incremental=incremental)


def fold_ops(ops_batches: list[list[dict[str, Any]]]) -> ContextState:
    """按时间顺序把若干批 op 折成一个当前状态（git 树状态那个意思）。"""
    state = ContextState()
    for ops in ops_batches:
        state.apply(ops)
    return state
