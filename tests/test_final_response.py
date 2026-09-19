"""final_response_text 的透传优先级测试。

回归：模型调用失败时错误曾被吞成 "(empty response)"——401 key 封锁、网关
多模态未开这类错误在聊天里完全不可见，操作者只能 SSH 查 journal。现在
完整透传 provider 错误，不截断不加工。
"""

from __future__ import annotations

from types import SimpleNamespace

from connectclaw.coding.coding_agent import final_response_text


def _result(content=(), error_message=None):
    return SimpleNamespace(content=list(content), error_message=error_message)


def test_text_blocks_win():
    r = _result(content=[{"type": "text", "text": "回答"}], error_message="残留错误")
    assert final_response_text(r) == "回答"


def test_text_blocks_join_newlines():
    r = _result(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert final_response_text(r) == "a\nb"


def test_empty_text_blocks_skipped():
    r = _result(content=[{"type": "text", "text": ""}])
    assert final_response_text(r) == "(empty response)"


def test_thinking_fallback():
    r = _result(content=[{"type": "thinking", "thinking": "只想没说"}])
    assert final_response_text(r) == "只想没说"


def test_error_passthrough_full():
    """stop_reason=error 时完整透传 API 错误——行动依据不能被吞。"""
    err = ("Error code: 401 - {'error': {'message': \"Authentication Error, "
           "Key is blocked. Update via `/key/unblock` if you're an admin.\", "
           "'type': 'auth_error', 'code': '401'}}")
    r = _result(content=[{"type": "text", "text": ""}], error_message=err)
    out = final_response_text(r)
    assert out.startswith("⚠️ 模型调用失败：")
    assert err in out  # 全文保留，不截断


def test_truly_empty_marker():
    assert final_response_text(_result()) == "(empty response)"
