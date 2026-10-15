"""Sandbox redesign regression tests (2026-09).

The redesign removed resource limits (RLIMIT_AS etc.) and per-network/auth
routing — the sandbox now only isolates, BashGuard gates danger, and bash
commands see the user's interactive PATH. These tests pin the new behavior.
"""

from __future__ import annotations

import asyncio
import os
import types

import connectclaw.coding.safety.sandbox as sandbox_mod
from connectclaw.coding.safety.sandbox import DirectSandbox
from connectclaw.coding.safety.shellpath import get_user_path, which_in_user_path, with_user_path
from connectclaw.coding.tools.bash import BashTool


# ── sandbox tier selection (runtime capability probing) ────────


def test_probe_results_are_bool():
    """The probes must answer a yes/no; they fork real (tiny) probe commands
    for bwrap / unshare and must never raise."""
    assert isinstance(sandbox_mod._probe_bwrap(), bool)
    assert isinstance(sandbox_mod._probe_unshare(), bool)


def test_detected_sandbox_can_run_a_command():
    """Environment-agnostic core regression: whatever tier probing picks, the
    returned sandbox must actually be able to execute a command. In
    namespace-restricted containers this falls back to Direct and must still
    work — a sandbox that can never run a command is worse than none."""
    cls = sandbox_mod.detect_best_sandbox()
    result = asyncio.run(cls(cwd="/tmp").execute("echo sandbox-selected-ok", timeout=15))
    assert result.exit_code == 0, result.stderr
    assert "sandbox-selected-ok" in result.stdout


def _force_which_present(monkeypatch) -> None:
    """Make shutil.which report every binary as present (installed), so the
    tier choice is driven purely by the probe cache below."""
    monkeypatch.setattr(
        sandbox_mod, "shutil",
        types.SimpleNamespace(which=lambda _name: "/usr/bin/probe"),
    )


def test_detect_falls_back_to_direct_when_namespaces_unavailable(monkeypatch):
    """Installed-but-unusable bwrap/unshare (probe fails — no permission to
    create namespaces) must NOT be selected: degrade to direct execution."""
    _force_which_present(monkeypatch)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "bwrap", False)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "unshare", False)
    assert sandbox_mod.detect_best_sandbox() is sandbox_mod.DirectSandbox


def test_detect_picks_bwrap_when_probe_succeeds(monkeypatch):
    """A namespace-capable environment keeps full isolation: bwrap wins over
    unshare over direct."""
    _force_which_present(monkeypatch)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "bwrap", True)
    assert sandbox_mod.detect_best_sandbox() is sandbox_mod.BwrapSandbox
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "bwrap", False)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "unshare", True)
    assert sandbox_mod.detect_best_sandbox() is sandbox_mod.NamespaceSandbox


def test_detect_skips_binary_that_is_not_installed(monkeypatch):
    """Missing binary short-circuits without even probing the tier."""
    monkeypatch.setattr(
        sandbox_mod, "shutil",
        types.SimpleNamespace(which=lambda _name: None),
    )
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "bwrap", True)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "unshare", True)
    assert sandbox_mod.detect_best_sandbox() is sandbox_mod.DirectSandbox


def test_user_namespace_failure_fingerprints():
    """The Codex fingerprint set is matched verbatim; unrelated text is not a ban."""
    for text in sandbox_mod.USER_NAMESPACE_FAILURES:
        assert sandbox_mod._is_user_namespace_failure(text), text
    assert not sandbox_mod._is_user_namespace_failure("bwrap: failed to parse argument")
    assert not sandbox_mod._is_user_namespace_failure("")


def test_probe_bwrap_semantics_follows_codex_fingerprints(monkeypatch):
    """bwrap downgrades ONLY on a confirmed user-namespace ban (Codex semantics):
    a confirmed ban → unusable; success → usable; any other failure → keep bwrap
    so a spurious probe failure does not cost us isolation."""
    monkeypatch.setattr(
        sandbox_mod, "_run_probe",
        lambda args, timeout=10: (False, "bwrap: No permissions to create a new namespace"),
    )
    monkeypatch.delitem(sandbox_mod._PROBE_CACHE, "bwrap", raising=False)
    assert sandbox_mod._probe_bwrap() is False

    monkeypatch.setattr(
        sandbox_mod, "_run_probe",
        lambda args, timeout=10: (True, ""),
    )
    monkeypatch.delitem(sandbox_mod._PROBE_CACHE, "bwrap", raising=False)
    assert sandbox_mod._probe_bwrap() is True

    monkeypatch.setattr(
        sandbox_mod, "_run_probe",
        lambda args, timeout=10: (False, "bwrap: failed to parse argument"),
    )
    monkeypatch.delitem(sandbox_mod._PROBE_CACHE, "bwrap", raising=False)
    assert sandbox_mod._probe_bwrap() is True


def test_probe_unshare_stays_strict(monkeypatch):
    """unshare fails with locale-dependent EPERM text no fingerprint covers, so
    any failed probe means the tier is unusable (strict exit-0 rule)."""
    monkeypatch.setattr(
        sandbox_mod, "_run_probe",
        lambda args, timeout=10: (False, "unshare: unshare 失败: 不允许的操作"),
    )
    monkeypatch.delitem(sandbox_mod._PROBE_CACHE, "unshare", raising=False)
    assert sandbox_mod._probe_unshare() is False

    monkeypatch.setattr(
        sandbox_mod, "_run_probe",
        lambda args, timeout=10: (True, ""),
    )
    monkeypatch.delitem(sandbox_mod._PROBE_CACHE, "unshare", raising=False)
    assert sandbox_mod._probe_unshare() is True


# ── shellpath ──────────────────────────────────────────────────


def test_user_path_is_nonempty():
    assert get_user_path()


def test_with_user_path_wraps_when_it_differs():
    wrapped = with_user_path("echo hi")
    if get_user_path() == os.environ.get("PATH", ""):
        assert wrapped == "echo hi"  # nothing to inject
    else:
        assert wrapped == f'export PATH="{get_user_path()}:$PATH"; echo hi'


def test_which_in_user_path_finds_common_bin():
    # sh is present on every PATH variant; the point is it resolves at all.
    assert which_in_user_path("sh") is not None


# ── sandbox: no rlimits, no network gating ─────────────────────


def test_direct_sandbox_runs_node_without_rlimit():
    """The core regression: node used to die on RLIMIT_AS=512MB inside the
    sandbox; with resource limits removed it must run."""
    async def run():
        return await DirectSandbox(cwd="/tmp").execute(
            'node -e "console.log(\'node-ok\')"', timeout=20
        )
    result = asyncio.run(run())
    assert result.exit_code == 0, result.stderr
    assert "node-ok" in result.stdout


def test_direct_sandbox_runs_plain_command():
    async def run():
        return await DirectSandbox(cwd="/tmp").execute("echo sandbox-ok", timeout=10)
    result = asyncio.run(run())
    assert result.exit_code == 0
    assert "sandbox-ok" in result.stdout


# ── bash tool surface ──────────────────────────────────────────


def test_bash_tool_has_no_network_or_unsandboxed_params():
    tool = BashTool(cwd="/tmp")
    props = tool.parameters["properties"]
    assert "allow_network" not in props
    assert "unsandboxed" not in props


# ── [sandbox] mode: operator-configured isolation tier (2026-10) ──


def test_sandbox_mode_aliases_normalize():
    from connectclaw.config import normalize_sandbox_mode as norm

    assert norm("") == "auto"
    assert norm("AUTO") == "auto"
    assert norm("bubblewrap") == "bwrap"
    assert norm("namespace") == "unshare"
    for spelling in ("none", "off", "false", "no", "direct", "unsandboxed", "full"):
        assert norm(spelling) == "none"
    assert norm("nonsense") == "auto"          # unknown never escalates


def test_resolve_mode_auto_follows_detection(monkeypatch):
    monkeypatch.setattr(sandbox_mod, "detect_best_sandbox",
                        lambda: sandbox_mod.NamespaceSandbox)
    assert sandbox_mod.resolve_sandbox_cls("auto") is sandbox_mod.NamespaceSandbox


def test_resolve_mode_none_is_direct_regardless_of_probes(monkeypatch):
    """mode=none is the operator's full-trust switch: no isolation, no probe."""
    _force_which_present(monkeypatch)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "bwrap", True)
    assert sandbox_mod.resolve_sandbox_cls("none") is sandbox_mod.DirectSandbox


def test_resolve_forced_mode_degrades_when_unusable(monkeypatch):
    """A forced tier must never make every command fail on a host that bans
    namespaces — it degrades to direct instead."""
    _force_which_present(monkeypatch)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "bwrap", False)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "unshare", False)
    assert sandbox_mod.resolve_sandbox_cls("bwrap") is sandbox_mod.DirectSandbox
    assert sandbox_mod.resolve_sandbox_cls("unshare") is sandbox_mod.DirectSandbox


def test_resolve_forced_mode_when_usable(monkeypatch):
    _force_which_present(monkeypatch)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "bwrap", True)
    assert sandbox_mod.resolve_sandbox_cls("bwrap") is sandbox_mod.BwrapSandbox
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "bwrap", False)
    monkeypatch.setitem(sandbox_mod._PROBE_CACHE, "unshare", True)
    assert sandbox_mod.resolve_sandbox_cls("unshare") is sandbox_mod.NamespaceSandbox


def test_config_reads_sandbox_mode(tmp_path, monkeypatch):
    from connectclaw.config import Config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[sandbox]\nmode = "none"\n')
    monkeypatch.delenv("CONNECTCLAW_SANDBOX", raising=False)
    assert Config.load(str(cfg_file)).sandbox.mode == "none"

    # env var wins over the file (deployments override without touching config)
    monkeypatch.setenv("CONNECTCLAW_SANDBOX", "off")
    assert Config.load(str(cfg_file)).sandbox.mode == "none"
    monkeypatch.setenv("CONNECTCLAW_SANDBOX", "auto")
    assert Config.load(str(cfg_file)).sandbox.mode == "auto"


def test_config_defaults_to_auto_without_section(tmp_path, monkeypatch):
    from connectclaw.config import Config

    monkeypatch.delenv("CONNECTCLAW_SANDBOX", raising=False)
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[agent]\ncwd = \"/tmp\"\n")
    assert Config.load(str(cfg_file)).sandbox.mode == "auto"


def test_bash_tool_uses_injected_sandbox_cls(tmp_path):
    """The [sandbox] mode plumbing reaches the agent's own bash tool."""
    tool = BashTool(cwd=str(tmp_path), sandbox_cls=DirectSandbox)
    assert tool._sandbox_cls is DirectSandbox
    result = asyncio.run(tool.execute("tc1", {"command": "echo injected-ok"}))
    assert "injected-ok" in result.content[0]["text"]
    assert result.details["sandbox_level"] == "direct"


def test_unsandboxed_mode_reaches_outside_cwd(tmp_path):
    """mode=none really means no isolation: bash writes outside the working
    directory instead of being confined by the bwrap binds."""
    proj = tmp_path / "proj"
    proj.mkdir()
    outside = tmp_path / "outside.txt"
    tool = BashTool(cwd=str(proj), sandbox_cls=DirectSandbox)
    asyncio.run(tool.execute("tc1", {"command": f"echo full-access > {outside}"}))
    assert outside.read_text().strip() == "full-access"


def test_coding_agent_applies_sandbox_mode():
    """[sandbox] mode reaches the agent's bash tool and stays agent-visible."""
    from connectclaw.coding.coding_agent import CodingAgent
    from connectclaw.config import Config, SandboxConfig

    cfg = Config()
    cfg.sandbox = SandboxConfig(mode="none")
    agent = CodingAgent(cfg)

    assert agent._bash_tool._sandbox_cls is DirectSandbox
    names = [t.name for t in agent._refresh_tools()]
    assert "bash" in names                      # still exposed, just not isolated
    facts = agent.environment_facts()
    assert "沙箱：已关闭" in facts                # the agent knows it is unjailed


