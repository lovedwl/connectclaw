"""Security-layer tests — the agent cannot touch escape-hatch config.

Covers: protected-path detection, write-hash_edit refusal, bwrap RO-bind args,
and the //bash operator gate (allow-list + BashGuard + auth) end-to-end at the
CodingAgent level.
"""

from __future__ import annotations

import asyncio

from connectclaw.coding.coding_agent import CodingAgent
from connectclaw.coding.safety.sandbox import BwrapSandbox
from connectclaw.coding.tools.hash_edit import create_hash_edit_tool
from connectclaw.coding.tools.write import create_write_tool
from connectclaw.config import BashConfig, Config

import connectclaw.security as security


def _patch_protected(monkeypatch, cfg_path: str) -> None:
    monkeypatch.setattr(security, "protected_file_paths", lambda: [cfg_path])


async def test_write_refuses_protected_file(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[bash]\n", encoding="utf-8")
    _patch_protected(monkeypatch, str(cfg))
    tool = create_write_tool(cwd=str(tmp_path), read_tool=None)
    res = await tool.execute("t1", {"file_path": str(cfg), "content": "hack"})
    assert "受保护" in res.content[0]["text"]
    assert res.details and res.details.get("is_error")
    assert cfg.read_text() == "[bash]\n"  # untouched


async def test_hash_edit_refuses_protected_file(tmp_path, monkeypatch):
    cfg = tmp_path / "models.toml"
    cfg.write_text("[[profiles]]\n", encoding="utf-8")
    _patch_protected(monkeypatch, str(cfg))
    tool = create_hash_edit_tool(cwd=str(tmp_path))
    res = await tool.execute("t1", {
        "path": str(cfg),
        "edits": [{"op": "replace_text", "oldText": "X", "newText": "Y"}],
    })
    assert res.details and res.details.get("is_error")
    assert "受保护" in res.content[0]["text"]


def test_is_protected_detects_absolute_and_relative():
    import os

    assert security.is_protected(os.path.expanduser("~/.connectclaw/config.toml")) is True
    assert security.is_protected(os.path.expanduser("~/.connectclaw/models.toml")) is True
    assert security.is_protected("/tmp/random.txt") is False


def test_bwrap_mounts_protected_files_read_only():
    # The args list must contain --ro-bind for config+registry inside the
    # writable home/cwd. (Build via a dry path: construct then inspect args is
    # internal; we assert at least that the builder references them.)
    assert any(f.endswith("config.toml") for f in security.protected_file_paths())
    assert any(f.endswith("models.toml") for f in security.protected_file_paths())


# ── operator bash gate ─────────────────────────────────────────


def _agent_with_operators(open_ids: list[str]) -> CodingAgent:
    cfg = Config()
    cfg.bash = BashConfig(operator_open_ids=open_ids)
    return CodingAgent(cfg)


async def test_operator_bash_denies_non_operator():
    ag = _agent_with_operators(["ou_owner"])
    r = await ag.run_operator_bash("oc", "ou_intruder", "echo hi")
    assert "权限" in r


async def test_operator_bash_empty_command_usage():
    ag = _agent_with_operators(["ou_owner"])
    r = await ag.run_operator_bash("oc", "ou_owner", "")
    assert "用法" in r


async def test_operator_bash_blocks_dangerous():
    ag = _agent_with_operators(["ou_owner"])
    r = await ag.run_operator_bash("oc", "ou_owner", "rm -rf /")
    assert "危险命令已拦截" in r


async def test_operator_bash_suspicious_needs_channel():
    ag = _agent_with_operators(["ou_owner"])
    r = await ag.run_operator_bash("oc", "ou_owner", "rm old.txt")
    # No channel wired → SUSPICIOUS can't be authorized → explicit refusal.
    assert "授权" in r and "rm old.txt" in r


async def test_operator_bash_safe_runs_directly():
    """Operator channel is unsandboxed by design (2026-10): safe commands run
    on the real host shell; security lives in the whitelist/DANGEROUS/
    SUSPICIOUS tests below and above."""
    ag = _agent_with_operators(["ou_owner"])
    r = await ag.run_operator_bash("oc", "ou_owner", "echo operator-ok")
    assert "operator-ok" in r
    # Codex JsonToolOutput 同款：结构化 JSON，exit_code 为固定字段
    assert '"exit_code": 0' in r


# ── /whoami self-discovery ─────────────────────────────────────


class FakeAgent:
    last_sender = "ou_fake_123"
    _config = Config()
    _config.bash = BashConfig(operator_open_ids=["ou_fake_123"])


async def test_whoami_reports_and_marks_whitelist():
    from connectclaw.commands import handle

    r = await handle("/whoami", conversation_key="oc", agent=FakeAgent())
    assert "ou_fake_123" in r
    assert "已在白名单" in r


async def test_whoami_marks_pending_when_not_listed():
    from connectclaw.commands import handle

    class No(Config):
        pass

    fa = FakeAgent()
    fa._config = Config()
    fa._config.bash = BashConfig(operator_open_ids=["ou_other"])
    r = await handle("/whoami", conversation_key="oc", agent=fa)
    assert "尚未加入" in r
    assert "operator_open_ids" in r


# ── `!`-prefix routing: media placeholders are NOT operator commands ──────


def test_image_placeholder_is_not_operator_bash():
    """SDK 把入站图片渲染成 `![image](img_v3_...)` —— 以 ! 开头但不是命令。

    回归：这类消息曾掉进 operator bash 分支被当 shell 执行，bash 报
    「未预期的记号」并把 stderr 刷进聊天。它必须走正常 agent 回合。
    """
    from connectclaw.main import is_operator_bash

    img = "![image](img_v3_0215k_fb65a176-562a-4014-8d2c-71273f0ecbag)"
    assert is_operator_bash(img) is False
    assert is_operator_bash("![表情](img_v3_xxx) 配文") is False
    assert is_operator_bash("") is False


def test_real_operator_lines_still_route_to_bash():
    from connectclaw.main import is_operator_bash

    assert is_operator_bash("!pwd") is True
    assert is_operator_bash("!ls -la") is True
    assert is_operator_bash("/bash echo hi") is True
    # `!` 单独一个字符不是命令（历史行为保留）
    assert is_operator_bash("!") is False
    # 普通消息不带 ! 前缀
    assert is_operator_bash("你好") is False
    # markdown 里非开头的图片引用不影响命令判定
    assert is_operator_bash("!cat a.md\n![image](img_x)") is True


# ── M16: BashGuard 结构化检查 ─────────────────────────────────

from connectclaw.coding.tools.bash import BashGuard  # noqa: E402


def test_bashguard_rm_flag_order_and_subshell():
    g = BashGuard()
    # 参数换序 / glob / 子 shell 包裹，此前全部漏判
    assert g.check("rm -fr /") == "DANGEROUS"
    assert g.check("rm -rf /*") == "DANGEROUS"
    assert g.check("(rm -rf /)") == "DANGEROUS"
    assert g.check("rm -rf $HOME") == "DANGEROUS"
    assert g.check("rm -rf ~") == "DANGEROUS"


def test_bashguard_redirect_sensitive_paths():
    g = BashGuard()
    assert g.check("echo x > /etc/passwd") == "DANGEROUS"
    assert g.check("cat key > ~/.ssh/authorized_keys") == "DANGEROUS"
    assert g.check("chmod -R 000 /") == "DANGEROUS"
    assert g.check("dd of=/dev/sda if=x") == "DANGEROUS"
    # 黑洞设备与普通文件不受影响
    assert g.check("echo hi > /dev/null") == "SAFE"
    assert g.check("echo hi > out.txt") == "SAFE"


def test_bashguard_benign_commands_still_safe():
    g = BashGuard()
    assert g.check("ls -la && git status") == "SAFE"
    assert g.check("grep foo /etc/passwd") == "SAFE"  # 读不拦
    assert g.check("rm -rf ./build") == "SUSPICIOUS"  # 普通删除仍走授权


def test_bashguard_crontab_truncate_suspicious():
    g = BashGuard()
    assert g.check("crontab -r") == "SUSPICIOUS"
    # truncate 命令本身走授权（结构性检查只覆盖 rm/chmod/重定向）
    assert g.check("truncate -s 0 /etc/passwd") == "SUSPICIOUS"
    assert g.check("truncate -s 0 app.log") == "SUSPICIOUS"
