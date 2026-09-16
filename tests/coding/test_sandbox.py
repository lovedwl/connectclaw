"""Sandbox redesign regression tests (2026-09).

The redesign removed resource limits (RLIMIT_AS etc.) and per-network/auth
routing — the sandbox now only isolates, BashGuard gates danger, and bash
commands see the user's interactive PATH. These tests pin the new behavior.
"""

from __future__ import annotations

import asyncio
import os

from connectclaw.coding.safety.sandbox import DirectSandbox
from connectclaw.coding.safety.shellpath import get_user_path, which_in_user_path, with_user_path
from connectclaw.coding.tools.bash import BashTool


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
