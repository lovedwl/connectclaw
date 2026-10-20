# vendor patches

已安装 wheel（site-packages）与 vendored 源码树里打的本地补丁。
`uv sync` / 重装依赖会把补丁冲掉，装完务必重新 apply 并跑一次回归测试。

| patch | 目标包 | 问题 |
|---|---|---|
| `lark_channel_merge_batch_keep_media_resources.patch` | lark-channel-sdk 1.1.0 | `merge_batch` 重建合并消息时丢 `content_text`/`resources`，与文本同批到达的文件/图片消息被整批静默丢弃（2026-10-20，tests/channel/test_sdk_merge_batch.py 覆盖） |

应用方法（对已装 wheel）：

```bash
cd .venv/lib/python3.14/site-packages
patch -p1 < ../../../../scripts/vendor_patches/lark_channel_merge_batch_keep_media_resources.patch
```
