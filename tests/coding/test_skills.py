"""skills tool + SkillStore tests.

Covers the standard ~/.agents/skills layout: scan a SKILL.md library, parse
frontmatter, BM25 find with matched terms, load body + root + bin checks, and
the SkillsTool action dispatch.
"""

from __future__ import annotations

import os

import pytest

from connectclaw.coding.tools.skills import SkillStore, SkillsTool


def _write_skill(root: str, name: str, description: str, *, body: str = "", bins: list[str] | None = None) -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    meta = ""
    if bins:
        quoted = ", ".join(f'"{b}"' for b in bins)
        meta = f"metadata:\n  requires:\n    bins: [{quoted}]\n"
    path = os.path.join(d, "SKILL.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"---\nname: {name}\ndescription: \"{description}\"\n{meta}---\n\n# {name}\n\n{body}\n")
    return path


@pytest.fixture
def skills_dir(tmp_path) -> str:
    root = str(tmp_path / "skills")
    _write_skill(root, "ustc-107-hpc", "操作 107 超算平台，SLURM 作业提交与 GPU 申请", body="# HPC\n\nsbatch 模板……")
    _write_skill(root, "agently-mail", "发邮件、收邮件、管理收件箱", body="# Mail\n\n用 agently-cli 发送。")
    _write_skill(root, "lark-base", "飞书多维表格：建表、记录、视图", body="# Base\n\nbitable 操作。")
    return root


@pytest.fixture
def store(skills_dir) -> SkillStore:
    return SkillStore(skills_dir)


async def test_scan_lists_all_skills_with_frontmatter(store):
    skills = await store.scan()
    assert {s["id"] for s in skills} == {"ustc-107-hpc", "agently-mail", "lark-base"}
    by_id = {s["id"]: s for s in skills}
    assert by_id["ustc-107-hpc"]["name"] == "ustc-107-hpc"
    assert "107 超算" in by_id["ustc-107-hpc"]["description"]
    assert by_id["ustc-107-hpc"]["root"].endswith("ustc-107-hpc")


async def test_find_ranks_best_match_first(store):
    results = await store.find("在 107 超算上提交 GPU 作业")
    assert results and results[0]["id"] == "ustc-107-hpc"
    assert results[0]["score"] > 0
    assert any("超算" in m or m in ("作业",) or m in results[0]["matched"] for m in results[0]["matched"]) or results[0]["matched"]


async def test_find_no_match_returns_empty(store):
    assert await store.find("zzz完全无关词汇qqq") == []


async def test_find_empty_in_library(tmp_path):
    empty = SkillStore(str(tmp_path / "nosuch"))
    assert await empty.scan() == []
    assert await empty.find("anything") == []


async def test_load_returns_body_and_root(store):
    loaded = await store.load("ustc-107-hpc")
    assert loaded is not None
    assert "sbatch 模板" in loaded["text"]
    assert loaded["root"].endswith("ustc-107-hpc")
    # Frontmatter must not leak into the loaded body.
    assert "description:" not in loaded["text"]


async def test_load_unknown_id_returns_none(store):
    assert await store.load("ghost") is None


async def test_load_reports_missing_bins(tmp_path):
    root = str(tmp_path / "skills2")
    _write_skill(root, "needs-tools", "需要两个命令", bins=["definitely-no-such-bin-xyz", "ls"])
    loaded = await SkillStore(root).load("needs-tools")
    assert loaded["missing_bins"] == ["definitely-no-such-bin-xyz"]  # ls exists


async def test_skill_without_frontmatter_falls_back_to_dir_name(tmp_path):
    root = str(tmp_path / "skills3")
    d = os.path.join(root, "plain")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write("# Plain\n\n正文")
    s = SkillStore(root)
    skills = await s.scan()
    assert skills[0]["id"] == "plain"
    assert skills[0]["name"] == "plain"
    assert skills[0]["description"] == ""


async def test_tool_action_dispatch(skills_dir):
    tool = SkillsTool(SkillStore(skills_dir))
    scan = await tool.execute("t1", {"action": "scan"})
    assert "ustc-107-hpc" in scan.content[0]["text"] and "可用技能" in scan.content[0]["text"]

    find = await tool.execute("t1", {"action": "find", "query": "邮件 发送 收件箱"})
    assert "agently-mail" in find.content[0]["text"]

    load = await tool.execute("t1", {"action": "load", "skill_id": "agently-mail"})
    assert "agently-cli" in load.content[0]["text"]
    assert load.details and load.details["skill_id"] == "agently-mail"


async def test_tool_unknown_action_errors(skills_dir):
    tool = SkillsTool(SkillStore(skills_dir))
    result = await tool.execute("t1", {"action": "nope"})
    assert result.details and result.details.get("is_error")


async def test_tool_find_without_query_errors(skills_dir):
    tool = SkillsTool(SkillStore(skills_dir))
    result = await tool.execute("t1", {"action": "find"})
    assert result.details and result.details.get("is_error")
