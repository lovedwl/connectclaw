"""Tests for background tasks (bash background=true + bg_tasks tool)."""
import asyncio

import pytest

from connectclaw.coding import bg_tasks as bg_mod
from connectclaw.coding.bg_tasks import BgTaskManager, BgTasksTool
from connectclaw.coding.safety.sandbox import BwrapSandbox, DirectSandbox
from connectclaw.coding.tools.bash import BashTool, BashGuard


@pytest.fixture(autouse=True)
def isolated_manager(tmp_path, monkeypatch):
    mgr = BgTaskManager(log_dir=str(tmp_path))
    monkeypatch.setattr(bg_mod, "_MANAGER", mgr)
    return mgr


def _text(res) -> str:
    return res.content[0]["text"]


# ── Sandbox.build（纯重构验收，不实际执行 bwrap——测试环境可能嵌套不可用）──

def test_bwrap_build_returns_full_argv():
    argv = BwrapSandbox(cwd="/tmp").build("echo hi")
    assert argv[0] == "bwrap"
    assert "--die-with-parent" in argv
    assert "--unshare-pid" in argv
    assert argv[-4:] == ["--", "bash", "-c", "echo hi"]


def test_direct_build():
    assert DirectSandbox(cwd="/tmp").build("echo hi") == ["/bin/bash", "-c", "echo hi"]


# ── Manager 生命周期 ──

@pytest.mark.asyncio
async def test_manager_exit_and_tail(isolated_manager):
    t = await isolated_manager.start(
        "echo hi", ["/bin/bash", "-c", "echo hi"], cwd="/tmp")
    for _ in range(40):
        if not t.running:
            break
        await asyncio.sleep(0.05)
    assert t.state == "exited" and t.exit_code == 0
    ok, text = isolated_manager.tail(t.task_id)
    assert ok and "hi" in text


@pytest.mark.asyncio
async def test_manager_kill_group(isolated_manager):
    t = await isolated_manager.start(
        "sleep 30", ["/bin/bash", "-c", "sleep 30"], cwd="/tmp")
    assert t.running
    ok, msg = await isolated_manager.kill(t.task_id)
    assert ok
    assert t.state == "killed"
    # 二次 kill 报「已经结束」
    ok2, _ = await isolated_manager.kill(t.task_id)
    assert not ok2


# ── BashTool background 分支（monkeypatch 到 DirectSandbox，避免嵌套 bwrap）──

@pytest.mark.asyncio
async def test_bash_background_flag(isolated_manager, monkeypatch, tmp_path):
    tool = BashTool(cwd=str(tmp_path), guard=BashGuard())
    monkeypatch.setattr(tool, "_sandbox_cls", DirectSandbox)
    res = await tool.execute("tc1", {"command": "sleep 30", "background": True})
    assert "background" in _text(res) and "t001" in _text(res)
    t = isolated_manager.get("t001")
    assert t is not None and t.running
    # bg_tasks 工具收尾
    bt = BgTasksTool()
    res2 = await bt.execute("tc2", {"action": "kill", "id": "t001"})
    assert "已终止" in _text(res2)
    assert t.state == "killed"


@pytest.mark.asyncio
async def test_background_still_guarded(tmp_path):
    tool = BashTool(cwd=str(tmp_path), guard=BashGuard())
    res = await tool.execute("tc1", {"command": "rm -rf /", "background": True})
    assert "危险" in _text(res) or "blocked" in _text(res).lower()
    assert bg_mod.get_bg_manager().list_tasks() == []  # 未启动任何任务


# ── bg_tasks list/tail 输出形状 ──

@pytest.mark.asyncio
async def test_bg_tasks_list_and_tail(isolated_manager):
    bt = BgTasksTool()
    res = await bt.execute("tc1", {"action": "list"})
    assert "没有后台任务" in _text(res)

    t = await isolated_manager.start(
        "echo hello-bg", ["/bin/bash", "-c", "echo hello-bg"], cwd="/tmp")
    for _ in range(40):
        if not t.running:
            break
        await asyncio.sleep(0.05)

    res = await bt.execute("tc2", {"action": "list"})
    assert "t001" in _text(res) and "exited" in _text(res)

    res = await bt.execute("tc3", {"action": "tail", "id": "t001"})
    assert "hello-bg" in _text(res)

    res = await bt.execute("tc4", {"action": "tail"})
    assert _text(res) == "tail 需要 id 参数"
    assert res.details.get("is_error") is True


# ── 工具曝光耦合：bg_tasks 跟着 bash 走（不改受保护的 [agent].tools）──

def test_bg_tasks_rides_along_with_bash():
    from connectclaw.coding.coding_agent import CodingAgent
    from connectclaw.config import Config

    agent = CodingAgent(Config())
    assert "bg_tasks" in agent._tool_registry          # 子 agent 可授权
    names = [getattr(t, "name", "") for t in agent._refresh_tools()]
    assert "bash" in names and "bg_tasks" in names
