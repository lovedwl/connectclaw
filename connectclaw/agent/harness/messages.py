"""Convert AgentMessage[] to LLM-compatible Message[]."""

import time

from connectclaw.provider.types import UserMessage

from ..types import AgentMessage


def convert_to_llm(messages: list[AgentMessage]) -> list:
    """
    Convert AgentMessage[] to LLM-compatible Message[].

    - bashExecution → user message with formatted output
    - compactionSummary → user message in <summary> tags
    - branchSummary → user message in <summary> tags
    - user/assistant/toolResult → pass through
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
