"""web_fetch plain-HTTP fast path: HTML → Markdown extraction.

The critical logic is the extraction (browser orchestration is thin and needs
real network), so tests target `_html_to_markdown` directly.
"""

from __future__ import annotations

from connectclaw.coding.tools.lightpanda import _html_to_markdown


def test_strips_script_and_style():
    html = "<html><head><style>a{color:red}</style></head><body>" \
           "<h1>Title</h1><p>Hello <b>world</b>.</p>" \
           "<script>alert('x')</script><p>After script</p></body></html>"
    md = _html_to_markdown(html)
    assert "alert" not in md
    assert "Hello world" in md
    assert "After script" in md


def test_title_headings_and_lists():
    html = ("<html><head><title>My Page</title></head><body>"
            "<h1>Intro</h1><p>First para.</p>"
            "<h2>Details</h2><ul><li>one</li><li>two</li></ul></body></html>")
    md = _html_to_markdown(html)
    assert md.startswith("# My Page")
    assert "\n# Intro" in md
    assert "## Details" in md
    assert "\n- one" in md
    assert "\n- two" in md


def test_links_kept_images_and_js_dropped():
    html = '<body><p>See <a href="https://example.com/docs">the docs</a> here.</p></body>'
    md = _html_to_markdown(html)
    assert "[the docs](https://example.com/docs)" in md
    # empty-text link (icon-only) and javascript hrefs produce no link syntax
    md2 = _html_to_markdown('<body><a href="https://x.com"><img src="a.png"></a></body>')
    assert "](https://x.com)" not in md2
    md3 = _html_to_markdown('<body><a href="javascript:void(0)">click</a></body>')
    assert "](javascript" not in md3


def test_page_chrome_skipped():
    html = ("<nav>menu bar</nav><header>site header</header>"
            "<main><p>real content</p></main>"
            "<footer>copyright 2020</footer><aside>ads ads</aside>")
    md = _html_to_markdown(html)
    assert "menu bar" not in md
    assert "site header" not in md
    assert "copyright" not in md
    assert "ads ads" not in md
    assert "real content" in md


def test_void_tags_do_not_leak_skip():
    # An unclosed void element (<meta> without </meta>) must not swallow the
    # rest of the page — the skip counter only counts container tags.
    html = '<head><meta charset="utf-8"><title>T</title></head><body><p>visible</p></body>'
    md = _html_to_markdown(html)
    assert "visible" in md


def test_inline_code_and_pre_blocks():
    html = "<p>Use <code>foo()</code> please.</p><pre><code>x = 1\ny = 2</code></pre>"
    md = _html_to_markdown(html)
    assert "`foo()`" in md
    assert "```\nx = 1\ny = 2\n```" in md


def test_simple_table():
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    md = _html_to_markdown(html)
    assert "| a | b" in md
    assert "| 1 | 2" in md


def test_whitespace_collapsed_and_entities():
    html = "<p>Fish &amp;   chips\n   with    spaces</p><p>Line2</p>"
    md = _html_to_markdown(html)
    assert "Fish & chips with spaces" in md
    assert "Line2" in md


def test_relative_links_resolved_against_base_url():
    html = ('<body><a href="docs/page.html">docs</a>'
            '<a href="https://cdn.other.com/x">ext</a>'
            '<a href="mailto:a@b.com">mail</a></body>')
    md = _html_to_markdown(html, base_url="https://example.com/start/")
    assert "[docs](https://example.com/start/docs/page.html)" in md
    assert "[ext](https://cdn.other.com/x)" in md
    assert "](mailto:" not in md  # non-http schemes keep the text, drop the link


def test_empty_list_items_dropped():
    html = "<ul><li><img src='i.png'></li><li>real</li></ul>"
    md = _html_to_markdown(html)
    assert md.strip() == "- real"
