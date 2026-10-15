"""2026-10 (H4) regression tests — web_fetch SSRF gate.

Covers:
- scheme allowlist (http/https only)
- loopback / private / link-local / metadata / unmapped-v4-in-v6 hosts rejected
- public hostnames pass
- the web_fetch tool refuses internal URLs before any network I/O
"""

from __future__ import annotations

import pytest

from connectclaw.security import check_public_http_url


@pytest.mark.parametrize("url", [
    # 公网 IP 字面量：不触发 DNS 解析，离线可测
    "https://93.184.216.34/page",
    "http://8.8.8.8/",
    "https://1.1.1.1/",
])
def test_public_url_allowed(url):
    assert check_public_http_url(url) is None


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:9222/json",
    "http://localhost/admin",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.1/",
    "http://192.168.1.1/",
    "http://172.16.0.1/",
    "http://[::1]/",
    "http://[::ffff:10.0.0.1]/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://0.0.0.0/",
])
def test_internal_url_rejected(url):
    reason = check_public_http_url(url)
    assert reason is not None


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/x",
    "javascript:alert(1)",
])
def test_non_http_scheme_rejected(url):
    reason = check_public_http_url(url)
    assert reason is not None
    assert "http/https" in reason


async def test_web_fetch_refuses_loopback_before_io(tmp_path):
    # 端口 1 上不会有服务；即便有也绝不能被访问——SSRF 校验必须先于 fetch
    from connectclaw.coding.tools.web_search import WebSearchConfig, WebFetchTool

    tool = WebFetchTool(config=WebSearchConfig(timeout=1))
    r = await tool.execute("t", {"url": "http://127.0.0.1:1/"})
    assert r.details.get("is_error") is True
    assert "被拒绝" in r.content[0]["text"]
