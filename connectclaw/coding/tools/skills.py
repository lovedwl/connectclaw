"""skills tool — read-only gateway to the machine's global skill library.

Skills are standard SKILL.md packages (the same ones ZCode uses): a directory
with a ``SKILL.md`` (frontmatter: name / description / ...) plus optional
support files/scripts. The main agent sees ONE ``skills`` tool and searches +
loads on demand, instead of every skill's text occupying the system prompt.

Protocol:
  scan  — list every skill (id + one-line description).
  find  — BM25 top-K over name+description, with matched terms, for a need.
  load  — return a skill's SKILL.md body (capped) + its root dir, so the model
          reads sibling skills / support files and runs its steps with its own
          read/bash tools. Loaded text stays a normal tool result in history —
          it never enters the system prompt, so the provider prefix cache is
          undisturbed.

Skills are reference documents: following their steps (bash, etc.) still goes
through the existing sandbox + authorization chain.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any

from connectclaw.agent.types import AgentTool, AgentToolResult
from connectclaw.logging import get_logger
from connectclaw.memory.bm25 import BM25Index

logger = get_logger(__name__)

DEFAULT_SKILLS_DIR = os.path.expanduser("~/.agents/skills")

_LOAD_MAX_CHARS = 8000        # SKILL.md body cap (mirrors web_fetch)
_SCAN_MAX_DESC = 160          # per-line description truncation
_FIND_TOP_K = 3

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*", re.S)
_KEY_LINE = re.compile(r"^([\w.-]+):\s*(.*)$")
_REQUIRED_BINS_RE = re.compile(r"bins\s*:\s*\[([^\]]*)\]", re.S)
_DISPLAY_TOK_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")


@dataclass
class _Skill:
    id: str
    name: str
    description: str
    root: str
    path: str


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Top-level scalar keys of a YAML-ish frontmatter block.

    Covers what real SKILL.md files use: single-line values (quoted or not) and
    indented continuation lines folded into the previous key. Nested structures
    (``metadata:``) are folded into their parent key and otherwise ignored;
    binary requirements are extracted separately by ``_required_bins``.
    """
    m = _FRONTMATTER.search(text)
    if not m:
        return {}
    data: dict[str, str] = {}
    cur_key: str | None = None
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        key_m = _KEY_LINE.match(line.strip())
        if key_m and not line[:1].isspace():
            cur_key = key_m.group(1)
            data[cur_key] = _strip_quotes(key_m.group(2))
        elif cur_key and line[:1].isspace():
            data[cur_key] += " " + _strip_quotes(line.strip())
    return data


def _required_bins(raw: str) -> list[str]:
    """Binary names from ``metadata.requires.bins`` (if present)."""
    m = _REQUIRED_BINS_RE.search(raw)
    if not m:
        return []
    return [tok.strip().strip('"').strip("'") for tok in m.group(1).split(",") if tok.strip()]


def _body(raw: str) -> str:
    """SKILL.md body — everything after the closing frontmatter fence."""
    m = _FRONTMATTER.match(raw)
    return raw[m.end():].strip() if m else raw.strip()


def _display_terms(text: str) -> list[str]:
    """Query terms for the "matched words" hint: latin words whole, CJK split
    into single chars (approximating BM25's own tokenization for display)."""
    terms: list[str] = []
    for tok in _DISPLAY_TOK_RE.findall(text or ""):
        if not tok:
            continue
        if tok[0].isascii():
            if tok not in terms:
                terms.append(tok)
        else:
            for ch in tok:
                if ch not in terms:
                    terms.append(ch)
    return terms


def _cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    note = f"\n…[技能正文超过 {max_chars} 字符已截断；可用 read 工具读取完整 SKILL.md]"
    return text[: max_chars - len(note)] + note


class SkillStore:
    """Lazy scan of a SKILL.md skill library, cached by file mtimes."""

    def __init__(self, skills_dir: str = DEFAULT_SKILLS_DIR):
        self._dir = skills_dir
        self._cache: list[_Skill] | None = None
        self._cache_sig: dict[str, float] | None = None
        self._lock = asyncio.Lock()

    @property
    def root(self) -> str:
        return self._dir

    def _scan_locked(self) -> list[_Skill]:
        sig: dict[str, float] = {}
        try:
            entries = sorted(os.listdir(self._dir))
        except OSError:
            entries = []
        for entry in entries:
            path = os.path.join(self._dir, entry, "SKILL.md")
            try:
                sig[path] = os.path.getmtime(path)
            except OSError:
                continue
        if self._cache is not None and sig == self._cache_sig:
            return self._cache

        items: list[_Skill] = []
        for path, mtime in sig.items():
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    raw = f.read()
            except OSError:
                continue
            _ = mtime  # content-addressed below; mtime is only our invalidation key
            root = os.path.dirname(path)
            name = os.path.basename(root)
            fm = _parse_frontmatter(raw)
            items.append(_Skill(
                id=name,
                name=fm.get("name") or name,
                description=fm.get("description", ""),
                root=root,
                path=path,
            ))
        self._cache = items
        self._cache_sig = sig
        return items

    def _to_dict(self, s: _Skill) -> dict:
        return {"id": s.id, "name": s.name, "description": s.description, "root": s.root}

    async def scan(self) -> list[dict]:
        async with self._lock:
            return [self._to_dict(s) for s in self._scan_locked()]

    async def find(self, query: str, top_k: int = _FIND_TOP_K) -> list[dict]:
        """Top-K skills by BM25 over name+description, with matched terms."""
        async with self._lock:
            skills = self._scan_locked()
        if not skills:
            return []
        scores = BM25Index([f"{s.name} {s.description}" for s in skills]).score(query or "")
        ranked = sorted(zip(skills, scores), key=lambda p: p[1], reverse=True)
        picked = [(s, sc) for s, sc in ranked if sc > 0][: max(1, top_k)]
        if not picked:
            return []
        terms = _display_terms(query)
        results = []
        for s, score in picked:
            index_text = f"{s.name} {s.description}".lower()
            matched = [t for t in terms if t.lower() in index_text]
            results.append({
                **self._to_dict(s),
                "score": round(score, 2),
                "matched": matched,
            })
        return results

    async def load(self, skill_id: str) -> dict | None:
        """Full body + root of a skill; None when id is unknown."""
        async with self._lock:
            skills = self._scan_locked()
        skill = next((s for s in skills if s.id == skill_id), None)
        if skill is None:
            return None
        try:
            with open(skill.path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
        except OSError:
            return None
        return {
            "id": skill.id,
            "name": skill.name,
            "root": skill.root,
            "text": _cap_text(_body(raw), _LOAD_MAX_CHARS),
            "missing_bins": [b for b in _required_bins(raw) if shutil.which(b) is None],
        }


class SkillsTool(AgentTool):
    name = "skills"
    label = "skills"
    description = (
        "按需检索/加载本机技能库（~/.agents/skills 下标准 SKILL.md 技能："
        "lark-*、agently-mail、ustc-107-hpc 等）。技能是参考流程文档——遵循其步骤仍走正常工具与授权。\n"
        "用法（action）：\n"
        "- scan：列出全部技能（id + 一行简介）\n"
        "- find：给需求描述，返回最匹配的 1-3 个技能（含命中词与依赖提示）\n"
        "- load：按 skill_id 载入技能全文（含技能目录；支持文件与脚本可再用 read/bash 处理）"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "find", "load"],
                "description": "要执行的操作",
            },
            "query": {
                "type": "string",
                "description": "find 用：需求/场景描述",
            },
            "skill_id": {
                "type": "string",
                "description": "load 用：技能 id（来自 scan/find）",
            },
            "top_k": {
                "type": "integer",
                "description": "find 用：返回条数（默认 3）",
            },
        },
        "required": ["action"],
    }

    def __init__(self, store: SkillStore):
        self._store = store

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        action = (params.get("action") or "").strip().lower()
        try:
            if action == "scan":
                return await self._exec_scan()
            if action == "find":
                return await self._exec_find(
                    (params.get("query") or "").strip(),
                    int(params.get("top_k") or _FIND_TOP_K),
                )
            if action == "load":
                return await self._exec_load((params.get("skill_id") or "").strip())
        except Exception as e:  # noqa: BLE001
            logger.warning("skills %s failed: %s", action, e)
            return self._err(f"skills {action} 执行失败: {e}")
        return self._err(f"未识别的 action: {action!r}（可用 scan / find / load）")

    # ── actions ───────────────────────────────────────────────

    async def _exec_scan(self) -> AgentToolResult:
        skills = await self._store.scan()
        if not skills:
            return AgentToolResult(
                content=[{"type": "text", "text": f"技能库为空（{self._store.root}）。"}],
            )
        lines = [f"可用技能（{len(skills)}）："]
        for s in skills:
            desc = s["description"].replace("\n", " ")[:_SCAN_MAX_DESC]
            lines.append(f"- id={s['id']}  {s['name']}：{desc}")
        lines.append("技能较多时建议用 find 按需检索最相关的。")
        return AgentToolResult(content=[{"type": "text", "text": "\n".join(lines)}])

    async def _exec_find(self, query: str, top_k: int) -> AgentToolResult:
        if not query:
            return self._err("find 需要 query（描述你要做的事/场景）。示例：find query=\"在 107 超算上跑深度学习作业\"")
        results = await self._store.find(query, top_k)
        if not results:
            return AgentToolResult(
                content=[{"type": "text",
                          "text": f"没有找到与“{query}”匹配的技能。可用 scan 查看全部，或换一个描述再试。"}],
            )
        lines = [f"与“{query}”最匹配的技能（top-{len(results)}）："]
        for i, r in enumerate(results, 1):
            desc = r["description"].replace("\n", " ")[:_SCAN_MAX_DESC]
            hit = "、".join(r["matched"]) if r["matched"] else "—"
            lines.append(f"{i}. [{r['score']}] id={r['id']} {r['name']}：{desc}")
            lines.append(f"   命中词: {hit} → load skill_id={r['id']} 载入")
        return AgentToolResult(
            content=[{"type": "text", "text": "\n".join(lines)}],
            details={"query": query},
        )

    async def _exec_load(self, skill_id: str) -> AgentToolResult:
        if not skill_id:
            return self._err("load 需要 skill_id（来自 scan/find）")
        loaded = await self._store.load(skill_id)
        if loaded is None:
            return AgentToolResult(
                content=[{"type": "text",
                          "text": f"找不到技能 id={skill_id}。先 scan 或 find 获取有效 id。"}],
            )
        lines = [
            f"技能：{loaded['name']}（{loaded['id']}）",
            f"目录：{loaded['root']}（支持文件可 read；步骤中的命令可 bash）",
        ]
        if loaded["missing_bins"]:
            lines.append(
                f"依赖检查：PATH 中未找到: {', '.join(loaded['missing_bins'])}——对应步骤可能无法执行"
            )
        lines.append("——————")
        lines.append(loaded["text"])
        return AgentToolResult(
            content=[{"type": "text", "text": "\n".join(lines)}],
            details={"skill_id": skill_id, "root": loaded["root"]},
        )

    @staticmethod
    def _err(msg: str) -> AgentToolResult:
        return AgentToolResult(content=[{"type": "text", "text": msg}], details={"is_error": True})


def create_skills_tool(skills_dir: str = DEFAULT_SKILLS_DIR) -> SkillsTool:
    return SkillsTool(SkillStore(skills_dir))
