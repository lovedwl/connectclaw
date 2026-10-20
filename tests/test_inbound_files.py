"""Regression: inbound Feishu file messages must land on disk.

Bug (2026-10-20): main.py only downloaded resources with type == "image";
type == "file" attachments (zip/pdf/…) were dropped and the agent only saw
the raw ``<file key=.../>`` placeholder text.
"""

import os

import pytest

from connectclaw import main as main_mod


class _Res:
    def __init__(self, type_, key, name=None):
        self.type = type_
        self.file_key = key
        self.file_name = name


class _FakeChannel:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.calls = []

    async def download_resource(self, file_key, resource_type="image", message_id=""):
        self.calls.append((file_key, resource_type, message_id))
        return self.payload


@pytest.mark.asyncio
async def test_file_resource_is_saved_and_annotated(tmp_path):
    ch = _FakeChannel(b"PK\x03\x04zipzipzip")
    res = [_Res("file", "file_v3_key1", "my game.zip")]

    saved = await main_mod._download_feishu_files(
        channel=ch,
        base_dir=str(tmp_path),
        conversation_key="oc_test",
        resources=res,
        message_id="om_1",
    )

    assert len(saved) == 1
    item = saved[0]
    assert os.path.isabs(item["path"])
    assert os.path.exists(item["path"])
    with open(item["path"], "rb") as f:
        assert f.read() == b"PK\x03\x04zipzipzip"
    # original name preserved for display, path stays shell-safe
    assert item["name"] == "my game.zip"
    assert " " not in os.path.basename(item["path"])
    # downloaded via the message-resource endpoint as a file
    assert ch.calls == [("file_v3_key1", "file", "om_1")]


@pytest.mark.asyncio
async def test_image_resources_are_not_touched_by_file_downloader(tmp_path):
    ch = _FakeChannel(b"x")
    saved = await main_mod._download_feishu_files(
        channel=ch,
        base_dir=str(tmp_path),
        conversation_key="oc_test",
        resources=[_Res("image", "img_key")],
        message_id="om_1",
    )
    assert saved == []
    assert ch.calls == []


@pytest.mark.asyncio
async def test_repeat_download_is_idempotent(tmp_path):
    ch = _FakeChannel(b"same-bytes")
    res = [_Res("file", "k1", "a.bin")]
    a = await main_mod._download_feishu_files(
        channel=ch, base_dir=str(tmp_path), conversation_key="oc_test",
        resources=res, message_id="om_1",
    )
    b = await main_mod._download_feishu_files(
        channel=ch, base_dir=str(tmp_path), conversation_key="oc_test",
        resources=res, message_id="om_2",
    )
    assert a[0]["path"] == b[0]["path"]
    assert os.listdir(tmp_path) == [os.path.basename(a[0]["path"])]


@pytest.mark.asyncio
async def test_oversize_file_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(main_mod, "MAX_INBOUND_FILE_BYTES", 10)
    ch = _FakeChannel(b"way too big for the cap")
    saved = await main_mod._download_feishu_files(
        channel=ch, base_dir=str(tmp_path), conversation_key="oc_test",
        resources=[_Res("file", "k1", "big.zip")], message_id="om_1",
    )
    assert saved == []
    assert not os.listdir(tmp_path)


def test_filename_sanitizer_blocks_traversal():
    out = main_mod._safe_upload_name("../../etc/passwd")
    assert "/" not in out and ".." not in out and out.endswith("passwd")
    assert main_mod._safe_upload_name("") == "file"
    assert main_mod._safe_upload_name(None) == "file"
