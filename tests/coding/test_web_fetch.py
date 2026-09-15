"""web_fetch plain-HTTP fast path: HTML → visible text extraction.

The critical logic is the extraction (browser orchestration is thin and needs
real network), so tests target `_html_to_text` directly.
"""

from __future__ import annotations

from connectclaw.coding.tools.lightpanda import _html_to_text


def test_strips_script_and_style():
    html = "<html><head><style>a{color:red}</style></head><body>" \
           "<h1>Title</h1><p>Hello <b>world</b>.</p>" \
           "<script>alert('x')</script><p>After script</p></body></html>"
    text = _html_to_text(html)
    assert "alert" not in text
    assert "Hello world" in text
    assert "After script" in text
    assert "Title" in text


def test_skips_noscript_and_template():
    html = "<noscript>enable JS</noscript><template><p>hidden</p></template><main>real content</main>"
    text = _html_to_text(html)
    assert "enable JS" not in text
    assert "hidden" not in text
    assert "real content" in text


def test_decodes_entities_and_collapses_whitespace():
    html = "<p>Fish &amp; chips</p><p>Line2</p>"
    text = _html_to_text(html)
    assert "Fish & chips" in text
    # Block tags separate into lines; runs of blank lines collapse to one.
    assert "\n" in text
