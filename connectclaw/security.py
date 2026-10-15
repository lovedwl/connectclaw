"""Paths the agent must NEVER modify — the escape-hatch security config.

The ``!``/``/bash`` operator whitelist lives in config.toml (``[bash]
operator_open_ids``) and the model registry in models.toml; those two files are
the lock of the escape hatches. The agent can write files (write / hash_edit)
and run bash, so both layers are closed here:
  - write / hash_edit refuse the paths outright;
  - the bwrap sandbox mounts them read-only even inside the writable cwd/home,
    so bash cannot touch them either.

Anything that would let the agent widen its own whitelist voids the escapes.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

from connectclaw.config import _find_config_path
from connectclaw.model_registry import DEFAULT_REGISTRY


def protected_file_paths() -> list[str]:
    """Absolute paths the agent may not read-write (config + registry)."""
    out: list[str] = []
    for p in (_find_config_path(), DEFAULT_REGISTRY):
        if p:
            out.append(os.path.realpath(os.path.abspath(os.path.expanduser(p))))
    return out


def is_protected(path: str) -> bool:
    # realpath（非 normpath/abspath）：不解析符号链接的话，agent 可以
    # `ln -s ~/.connectclaw ./link` 后经 ./link/config.toml 绕过判定（H2c）。
    ap = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    return any(os.path.realpath(p) == ap for p in protected_file_paths())


def resolve_tool_path(cwd: str, path: str) -> str:
    """Resolve a tool path against cwd; normalize away ``..``/``.``.

    绝对路径也必须 normpath：否则 ``/proj/../proj2/f`` 能以 ``startswith("/proj/")``
    绕过 cwd 前缀检查与 is_protected（M24）。
    """
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(os.path.abspath(cwd), path))


def path_within_cwd(cwd: str, absolute_path: str) -> bool:
    """True if absolute_path resolves inside cwd (symlinks resolved)."""
    cwd_real = os.path.realpath(os.path.abspath(cwd))
    p_real = os.path.realpath(absolute_path)
    return p_real == cwd_real or p_real.startswith(cwd_real + os.sep)


# ── SSRF gate（H4）─────────────────────────────────────────


def _ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # IPv4-mapped IPv6（::ffff:10.0.0.1）归一成 v4 再判，否则绕过 is_private
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _host_blocked(host: str) -> bool:
    """True for loopback / private / link-local / metadata targets."""
    host = host.strip().lower().rstrip(".")
    if not host:
        return True
    if (
        host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".internal")
        or host.endswith(".local")
    ):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # 域名：解析后任一地址命中即拒绝（DNS rebinding 的保守侧）
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except OSError:
            return True  # 解析不了 → 保守拒绝
        addrs = {i[4][0] for i in infos}
        if not addrs:
            return True
        try:
            return any(_ip_blocked(ipaddress.ip_address(a)) for a in addrs)
        except ValueError:
            return True
    return _ip_blocked(ip)


def check_public_http_url(url: str) -> str | None:
    """SSRF 校验：scheme ∈ {http, https} 且 host 非本机/内网/链路本地/元数据。

    返回 None 表示放行；否则返回拒绝理由。**DNS 解析是阻塞调用**，
    async 调用方应放进线程池（asyncio.to_thread）。
    注意：这只覆盖入口 URL；重定向后的最终 URL 由各 fetch 路径自行复查。
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return "URL 无法解析"
    if parsed.scheme not in ("http", "https"):
        return f"仅允许 http/https URL（收到 scheme={parsed.scheme or '(空)'}）"
    host = parsed.hostname
    if not host:
        return "URL 缺少 host"
    if _host_blocked(host):
        return f"禁止访问本机/内网/链路本地/元数据地址：{host}"
    return None
