"""Regression: Feishu `post` messages may carry attachments in a TOP-LEVEL
`files[]` array (observed 2026-10-20 when forwarding multiple files in one
message: content=[[]] empty, files[] holds the payloads).

The post converter only scanned paragraph elements, so every such message
normalized to empty text + zero resources and was dropped as "empty inbound"
— the forwarded 80MB game-APK parts vanished with no trace.
"""

import json

from lark_channel.channel.normalize.registry import parse_message_content
from lark_channel.channel.normalize.converters.post import convert

# Verbatim shape captured from the live [raw] event log (2026-10-20 16:39:35).
POST_CONTENT = json.dumps({
    "title": "",
    "content": [[]],
    "content_v2": [[]],
    "files": [
        {"file_key": "file_v3_aaa", "file_name": "base(1).apk.1", "is_folder": False},
        {"file_key": "file_v3_bbb", "file_name": "base(2).apk.2", "is_folder": False},
    ],
})


def test_post_top_level_files_become_file_resources():
    content = parse_message_content("post", POST_CONTENT)
    _text, resources = convert(content)
    got = {(r.type, r.file_key, r.file_name) for r in resources}
    assert got == {
        ("file", "file_v3_aaa", "base(1).apk.1"),
        ("file", "file_v3_bbb", "base(2).apk.2"),
    }


def test_post_folder_entries_not_surfaced():
    content = parse_message_content("post", json.dumps({
        "content": [[]],
        "files": [{"file_key": "fk", "file_name": "dir", "is_folder": True}],
    }))
    _text, resources = convert(content)
    assert resources == []


def test_post_inline_and_top_level_files_deduped():
    content = parse_message_content("post", json.dumps({
        "content": [[{"tag": "file", "file_key": "fk1", "file_name": "x.zip"}]],
        "files": [{"file_key": "fk1", "file_name": "x.zip", "is_folder": False}],
    }))
    _text, resources = convert(content)
    assert [(r.type, r.file_key) for r in resources] == [("file", "fk1")]
