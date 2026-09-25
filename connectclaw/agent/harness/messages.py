"""Convert AgentMessage[] to LLM-compatible Message[].

**这里绝不能改写历史消息。** 注入上下文（记忆 / agents 清单）是随用户消息一起
落盘、并作为下一轮的固定前缀存在的——上游按请求前缀缓存。一旦在这里把历史里的
注入块降级或抹掉，前缀就正好在被改写的那一轮断开，之后每轮都要多付一整轮的
未命中（2026-09-24 实测：平均命中率 77% → 61%，短会话 71% → 10%）。

窗口要干净，靠的是**结构化 + 按需注入**：注入是独立的 `context` 条目（op），只在内容
真的变化时才产出新的 op，历史保持纯追加（见 connectclaw/injection.py）。不要走"事后
改写历史"这条路。
"""

import time

from connectclaw.provider.types import UserMessage

from ..types import AgentMessage


def convert_to_llm(messages: list[AgentMessage]) -> list:
    """
    Convert AgentMessage[] to LLM-compatible Message[].

    - bashExecution → user message with formatted output
    - compactionSummary → user message in <summary> tags
    - branchSummary → user message in <summary> tags
    - user/assistant/toolResult → pass through unchanged
    """
    results = []
    for m in messages:
        role = m.role

        if role == "bashExecution":
            ts = m.timestamp or time.time() * 1000
            text = f"<bash-output command=\"{m.command}\">\n{m.output}\n</bash-output>"
            results.append(UserMessage(content=text, timestamp=ts))

        elif role == "compactionSummary":
            ts = m.timestamp or time.time() * 1000
            text = f"<summary>\n{m.summary}\n</summary>"
            results.append(UserMessage(content=text, timestamp=ts))

        elif role == "branchSummary":
            ts = m.timestamp or time.time() * 1000
            text = f"<summary>\n{m.summary}\n</summary>"
            results.append(UserMessage(content=text, timestamp=ts))

        elif role in ("user", "assistant", "toolResult"):
            results.append(m)

    return results
