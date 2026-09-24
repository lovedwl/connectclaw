"""web 抓取路径的内存边界（2026-09-25 OOM 事故的护栏）。

事故：一次 web_search 后 unit 峰值 **15.5G**，内核 OOM 杀掉、整机 swap 抖动。那次运行
4 次 Lightpanda 抓取全失败。这个文件里的三处无界缓冲是嫌疑：CDP 单条消息无上限、
页面文本全量进缓存（只按条数封顶）、共享浏览器子进程无上限无观测。这里把边界钉住。
"""

from __future__ import annotations

import inspect

from connectclaw.coding.tools import lightpanda as lp


def test_page_text_is_capped():
    long_page = "x" * (lp._MAX_PAGE_CHARS + 1234)
    assert len(lp._cap_page(long_page)) == lp._MAX_PAGE_CHARS
    assert lp._cap_page("短页面") == "短页面"


def test_fetch_cache_evicts_on_total_chars_not_only_count():
    lp._fetch_cache.clear()
    try:
        half = "x" * (lp._FETCH_CACHE_MAX_CHARS // 2)
        lp._cache_put("a", half)
        lp._cache_put("b", half)
        lp._cache_put("c", half)          # 总量 1.5× 上限 → 必须淘汰
        assert lp._cache_chars() <= lp._FETCH_CACHE_MAX_CHARS
        assert len(lp._fetch_cache) < 3
        assert "c" in lp._fetch_cache, "最新一条必须留下"
    finally:
        lp._fetch_cache.clear()


def test_cdp_websocket_has_a_message_size_cap():
    """不能让 CDP 单条消息无上限——整页 DOM 会一次性读进 Python 进程。"""
    source = inspect.getsource(lp.LightpandaEngine)
    assert "max_size=None" not in source
    assert "max_size=_CDP_MAX_MESSAGE_BYTES" in source


def test_child_memory_watchdog_exists():
    assert lp._CHILD_RSS_KILL_MB > lp._CHILD_RSS_WARN_MB > 0
    # 没有子进程时不能炸
    assert lp._child_rss_mb() == 0.0
    lp._check_child_memory()
