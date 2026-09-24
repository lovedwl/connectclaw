"""当轮注入上下文的统一登记处。

每轮对话前，ConnectClaw 会把三类「当轮上下文」拼在**用户消息之前**：

    <remembered-context>…</remembered-context>         记忆（persona + 按需召回）
    <retrieved-documents>…</retrieved-documents>        RAG 文档
    ## 可运行的 agents …（+ ## 可授权给子 agent 的工具 …）  子 agent / 工具清单

放在用户消息而不是 system prompt 里，是为了让 system prompt 与 tools 数组保持
字节稳定、让上游的 prompt cache 一直命中（见 coding_agent 里 prompt() 的注释
与 tools/agents.py 顶部关于 agents 元工具的说明）。

代价是这些块会随消息一起落盘：历史里每一轮都留一份副本。实测 168 个用户轮
全部带记忆块、同一句话平均重复注入 34 次，而且配置类旧说法（如「RAG 已启用」）
会出现 84 次、当天写的更正只出现 1 次——旧副本还会与新事实同框冲突。

**注意：剥离历史副本本身是要付缓存代价的**（2026-09-24 实测，别把这里当成"顺手
优化"）：该上游是**严格前缀缓存**（同一前缀第二次请求 cached_tokens=7168/7235，
改开头即失效、改结尾仍命中）。老设计之所以前缀缓存近乎全中，靠的正是"注入垃圾
被冻进历史、永不重写"这个副产品。而把**上一轮**那份也剥掉，会让分歧点落在上一轮
的用户消息上 → 那一整轮（消息 + 回复 + 工具结果）都变成未命中。逐轮模拟：平均命中
率 77% → 61%，短会话（6 轮）71% → 10%。

要同时拿到"干净窗口"和"缓存几乎不掉"，正确做法是**注入不落盘、每轮作为末尾
追加块发送**（只丢上一轮那 1.6k 注入本身，而不是整轮）。当前实现是本模块 +
``messages.convert_to_llm`` 的"历史副本降级"方案，属于退而求其次。

本模块只负责**认出**这些块，供三处使用：

  * ``memory/retriever``、``rag/retriever``、``tools/agents`` —— 产出（格式自定）；
  * ``agent/harness/messages.convert_to_llm`` —— 转成 LLM 上下文时，把**历史轮次**
    里的这些块降级成常量占位符（最近一条用户消息里那份保留，当轮要用）；
  * ``agent/harness/compaction._serialize`` —— 压缩转录时同样剥掉，避免过时记忆
    被写进摘要（摘要会长期留在上下文里）。

新增一类注入内容时，只要在这里加一条 pattern，不要在别处各写一遍。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# 剥掉历史注入块后留下的占位符。用常量是为了让"被剥掉的那些轮"彼此一致——
# 它们仍然与当年实际发出的字节不同，所以缓存仍在那一点断开（见上面的实测）。
INJECTION_PLACEHOLDER = "[当轮注入上下文已省略：记忆 / 文档 / agents 清单]"

# 「## 小节」型注入没有结束标记，按「标题 + 直到空行为止的行」界定。
# 行尾允许是换行**或字符串结尾**——末行常常没有换行，只写 \n 会把最后一行漏掉。
_SECTION_LINES = r"(?:(?!\n)[^\n]+(?:\n|\Z))*"
_AGENTS_HEAD = r"##\s*可运行的 agents[^\n]*(?:\n|\Z)"
_TOOLS_HEAD = r"##\s*可授权给子 agent 的工具[^\n]*(?:\n|\Z)"

# 每条 pattern 都从字符串开头匹配一段注入内容（含其后的空行分隔）。
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\A\s*<remembered-context>.*?</remembered-context>", re.S),
    re.compile(r"\A\s*<retrieved-documents>.*?</retrieved-documents>", re.S),
    re.compile(r"\A\s*" + _AGENTS_HEAD + _SECTION_LINES),
    re.compile(r"\A\s*" + _TOOLS_HEAD + _SECTION_LINES),
)


def split_injections(text: str) -> tuple[str, str]:
    """把开头的注入段与正文分开，返回 ``(注入段, 正文)``。

    只认开头的连续注入段。没有任何注入时原样返回 ``("", text)``，不做任何
    空白处理——避免动到用户正文里的缩进。
    """
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


def strip_injections(text: str) -> str:
    """把开头的注入段换成常量占位符（没有注入时原样返回）。

    幂等：对结果再调一次不会变。
    """
    if not text or text.startswith(INJECTION_PLACEHOLDER):
        return text
    injected, rest = split_injections(text)
    if not injected:
        return text
    return f"{INJECTION_PLACEHOLDER}\n\n{rest}" if rest else INJECTION_PLACEHOLDER


def has_injections(text: str) -> bool:
    return bool(text) and bool(_PATTERNS[0].match(text) or _PATTERNS[2].match(text) or _PATTERNS[3].match(text))


def strip_message_content(content: object) -> object:
    """给 ``str`` 或 ``list[dict]`` 形式的 content 做剥离（用户消息两种形状都有）。

    列表形状只在文本块上替换，图片等其它块原样保留。
    """
    if isinstance(content, str):
        return strip_injections(content)
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if _PATTERNS[0].match(text) or _PATTERNS[2].match(text) or _PATTERNS[3].match(text):
                    out.append({**block, "text": strip_injections(text)})
                    continue
            out.append(block)
        return out
    return content


# ── 按需注入（增量账本）─────────────────────────────────────────
#
# 用户 2026-09-24 拍板的策略：**只在真变化时才注入** —— 记忆多召回/少召回、
# agents 清单改动，都是"变化"；没变就别再发一遍。这样：
#   * 历史保持纯追加 → 前缀缓存不会因为改写历史而失效（见模块开头的实测）；
#   * 同一句话不再被重复注入 N 次（老设计实测平均 34 次）；
#   * 过时副本不再一轮轮堆积（老设计里「RAG 已启用」出现过 84 次）。
# 代价是模型要靠历史里更早的那次注入来回想，因此**未被提到的按沿用处理**，
# 注入块的表头会写明这一点。


@dataclass
class InjectionLedger:
    """一个会话的「已经注入过什么」账本。"""

    # memory_id -> {"line": 上次注入的渲染文本, "content": 原始内容}
    memory: dict[str, dict[str, str]] = field(default_factory=dict)
    # 上次注入的 agents/工具清单原文
    catalog: str | None = None

    def memory_delta(
        self,
        items: list[tuple[str, str, str]],
        alive: Callable[[str], bool] | None = None,
    ) -> tuple[list[str], list[str]]:
        """``items = [(memory_id, 渲染文本, 原始内容)]``。

        返回 ``(本次要注入的行, 需要告知"已遗忘"的行)``。**只按"内容"比对**：
        新增的、内容被更正过的算增量；内容没变的**不再重复注入**。

        为什么不用渲染文本当比对键：渲染文本里带着强度值，而强度每轮都可能被
        ``confirm_usage`` / 衰减改动——拿它比对的话，每条注入过的记忆都会每轮被判成
        "变了"，增量机制直接失效（第一版就是这么错的，真实会话回放能看出来）。强度
        的小幅变化对模型没有意义，不值得为此重发一遍。

        传了 ``alive`` 时，还会检查"注入过但现在库里没有了"的条目（``/forget`` 软删
        也算没了），产出撤销行——这就是"少召回"那一侧的告知。
        """
        current = {mid: (line, content) for mid, line, content in items}

        new_lines: list[str] = []
        for mid, (line, content) in current.items():
            prev = self.memory.get(mid)
            if prev is None or prev.get("content") != content:
                new_lines.append(line)
                self.memory[mid] = {"line": line, "content": content}

        forgotten: list[str] = []
        if alive is not None:
            for mid in list(self.memory):
                if mid in current:
                    continue
                if not alive(mid):
                    content = self.memory.pop(mid).get("content", "")
                    if content:
                        forgotten.append(f"- 已遗忘：{content}")

        return new_lines, forgotten

    def catalog_delta(self, text: str) -> str:
        """清单变了才返回；首次调用也算"变化"（新会话要先交代一遍）。"""
        if not text or text == self.catalog:
            return ""
        self.catalog = text
        return text


class LedgerRegistry:
    """按会话（用 session id——换会话/``/new`` 会换它）存取账本。

    新会话 = 新账本 = 首轮完整注入一次，语义正好；进程重启后也一样（保守地重发
    一次全量，不会漏）。
    """

    def __init__(self) -> None:
        self._by_session: dict[str, InjectionLedger] = {}

    def for_session(self, session_id: str) -> InjectionLedger:
        ledger = self._by_session.get(session_id)
        if ledger is None:
            ledger = self._by_session[session_id] = InjectionLedger()
        return ledger

    def drop(self, session_id: str) -> None:
        """会话结束/被丢弃时清账本（例如压缩重写了历史，需要重新完整注入）。"""
        self._by_session.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._by_session)


# ── 压缩时 squash 注入（像 git 合并历史）────────────────────────
#
# 按需注入让历史里散落着若干"增量块"。压缩会把一整段历史替换成摘要——如果那些
# 增量块跟着消失，模型就丢了记忆，而账本还以为它们"已经在上下文里"。所以压缩时
# 把区域内的注入块**合并成一个当前状态块**（git squash 语义），随摘要一起进入
# 上下文：模型不丢记忆，账本继续成立（不必重发），下一个压缩还能继续合并它。

_MERGED_HEADER = (
    "(压缩后合并的记忆状态：以下是此前注入过的记忆的当前值。"
    "每条前缀 [日期 · 强度]：日期是记录时间，强度 0~1 是可信度)"
)

_MEMORY_BLOCK_SEARCH = re.compile(r"<remembered-context>(.*?)</remembered-context>", re.S)
_CATALOG_SEARCH = re.compile(
    _AGENTS_HEAD + _SECTION_LINES + r"(?:\n*" + _TOOLS_HEAD + _SECTION_LINES + r")?",
    re.M,
)
_FORGOTTEN_PREFIX = "已遗忘："


def _memory_lines(text: str) -> list[str]:
    """取出注入块里的条目行（Detail 续行并入上一条）。"""
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


def line_content(line: str) -> str:
    """去掉 ``- [日期 · 强度]`` 与 ``[past]/[pattern]`` 前缀，得到内容本身。

    内容即"同一条记忆"的比对键（与账本 ``memory_delta`` 的比对口径一致）。
    """
    text = re.sub(r"^-\s+", "", line or "")
    text = re.sub(r"^\[[^\]]*\]\s*", "", text)          # [日期 · 强度]
    text = re.sub(r"^\[(?:past|pattern)\]\s*", "", text)  # 类型标记
    return text.split("\n")[0].strip()


def merge_injections(texts: list[str], *, header: str | None = None) -> str:
    """把若干注入块合并成一个**当前状态**块（git squash 语义）。

    按时间顺序重放：同内容的条目以最后一次出现为准；``- 已遗忘：X`` 把 X 从状态里
    移除（等于后来的提交 revert 了先前的）。agents/工具清单取最后一个版本。

    返回的块自身也是标准注入块，因此下次压缩可以继续合并它。
    """
    state: dict[str, str] = {}
    catalog = ""

    for text in texts:
        if not text:
            continue
        for line in _memory_lines(text):
            content = line_content(line)
            if not content:
                continue
            if content.startswith(_FORGOTTEN_PREFIX):
                target = " ".join(content[len(_FORGOTTEN_PREFIX):].split())
                for key in list(state):
                    if " ".join(key.split()) == target:
                        state.pop(key)
                continue
            state[content] = line
        match = _CATALOG_SEARCH.search(text)
        if match:
            catalog = match.group(0).strip()

    blocks: list[str] = []
    if state:
        blocks.append("\n".join(
            ["<remembered-context>", header or _MERGED_HEADER, *state.values(), "</remembered-context>"]
        ))
    if catalog:
        blocks.append(catalog)
    return "\n\n".join(blocks)
