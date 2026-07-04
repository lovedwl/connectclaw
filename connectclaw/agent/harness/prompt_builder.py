"""
Lightweight system prompt builder — pi-mono style.

Minimal base prompt + skills injected as XML blocks.
Tools are described by the LLM function-calling schema, not in the prompt.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

from connectclaw.logging import get_logger

logger = get_logger(__name__)


# ── Minimal default template ───────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are ConnectClaw, an AI assistant with access to tools.

## Environment
{cwd} | {date} | {os} | {shell}

## Rules
- Read files before writing to them. Use absolute paths.
- Commands run in a sandbox (bwrap): filesystem read-only except {cwd}, network blocked.
- If a command fails due to sandbox, retry with `allow_network: true`.
- Be concise and direct."""


class PromptBuilder:
    """Builds the system prompt for each turn.

    Usage:
        builder = PromptBuilder(cwd="/project")
        prompt = builder.build(rag_context="...", skills=[...])
    """

    def __init__(self, cwd: str, template_path: str | None = None):
        self.cwd = cwd
        self._template_path = template_path or os.path.expanduser(
            "~/.connectclaw/prompts/system.md"
        )
        self._template = self._load_template()

    def _load_template(self) -> str:
        for path in [
            self._template_path,
            str(Path(__file__).parent / "prompts" / "system.md"),
        ]:
            try:
                p = Path(path)
                if p.exists():
                    return p.read_text(encoding="utf-8")
            except Exception as e:
                logger.debug("Failed to load prompt template %s: %s", path, e)
        return DEFAULT_SYSTEM_PROMPT

    def build(
        self,
        *,
        rag_context: str = "",
        skills: list[dict[str, str]] | None = None,
    ) -> str:
        """Assemble the prompt: base template + skills (XML) + RAG context."""
        prompt = self._template.format(
            cwd=self.cwd,
            date=time.strftime("%Y-%m-%d"),
            os=platform.system(),
            shell=os.environ.get("SHELL", "/bin/bash"),
        )

        if skills:
            prompt += "\n\n" + _format_skills(skills)

        if rag_context:
            prompt += "\n\n" + rag_context

        return prompt

    def reload_template(self) -> None:
        self._template = self._load_template()


# ── Skills formatting (pi-mono XML style) ──────────────────────


def _format_skills(skills: list[dict[str, str]]) -> str:
    """Format skills as XML block, pi-mono style."""
    visible = [s for s in skills if not s.get("disable_model_invocation")]
    if not visible:
        return ""

    lines = [
        "The following skills provide specialized instructions.",
        "Read the full skill when the task matches its description.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.get('name', ''))}</name>")
        lines.append(f"    <description>{_escape_xml(skill.get('description', ''))}</description>")
        if skill.get("file_path"):
            lines.append(f"    <location>{_escape_xml(skill['file_path'])}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
