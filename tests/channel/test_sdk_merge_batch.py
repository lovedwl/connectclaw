"""Regression: lark_channel merge_batch must not drop media/resources.

Bug (2026-10-20): a file message merged with a text message inside the
600ms debounce window was rebuilt as a fresh InboundMessage without
``content_text`` / ``resources`` — the channel layer then saw an empty
message and silently dropped the whole batch (zip sent together with a
comment vanished without a trace).
"""

from lark_channel.channel.safety.chat_pipeline import merge_batch
from lark_channel.channel.types import (
    Conversation,
    FileContent,
    Identity,
    InboundMessage,
    ResourceDescriptor,
    TextContent,
)


def _msg(mid: str, content, content_text: str, resources=None) -> InboundMessage:
    return InboundMessage(
        id=mid,
        create_time=1,
        conversation=Conversation(chat_id="oc_test", chat_type="p2p"),
        sender=Identity(open_id="ou_user"),
        content=content,
        content_text=content_text,
        resources=resources or [],
        raw_content_type="file" if isinstance(content, FileContent) else "text",
    )


def test_merge_file_and_text_preserves_resources_and_text():
    file_msg = _msg(
        "om_file",
        FileContent(file_key="file_v3_abc", file_name="game.zip"),
        '<file key="file_v3_abc" name="game.zip"/>',
        [ResourceDescriptor(type="file", file_key="file_v3_abc", file_name="game.zip")],
    )
    text_msg = _msg("om_text", TextContent(text="放到tmp下"), "放到tmp下")

    merged = merge_batch([file_msg, text_msg])

    assert "game.zip" in merged.content_text
    assert "放到tmp下" in merged.content_text
    assert len(merged.resources) == 1
    assert merged.resources[0].type == "file"
    assert merged.resources[0].file_key == "file_v3_abc"
    # sources must remain reachable for consumers that iterate them
    assert len(merged.batched_sources or []) == 2


def test_merge_text_only_keeps_content_text():
    a = _msg("om_a", TextContent(text="第一行"), "第一行")
    b = _msg("om_b", TextContent(text="第二行"), "第二行")

    merged = merge_batch([a, b])

    assert "第一行" in merged.content_text
    assert "第二行" in merged.content_text


def test_merge_dedupes_identical_resources():
    res = ResourceDescriptor(type="image", file_key="img_1")
    a = _msg("om_a", TextContent(text="a"), "a", [res])
    b = _msg("om_b", TextContent(text="b"), "b", [
        ResourceDescriptor(type="image", file_key="img_1"),
    ])

    merged = merge_batch([a, b])

    assert len(merged.resources) == 1
