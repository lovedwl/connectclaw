"""Shared image helpers — MIME sniffing for inbound image landing.

Extracted from the retired image_analyze tool (superseded by inline
multimodal images + attach_image, see b376213). main.py uses these when
persisting downloaded Feishu images to the attachments dir.
"""

from __future__ import annotations


def detect_mime_type(image_data: bytes) -> str:
    """Detect MIME type from magic bytes. Returns 'image/png' as safe default."""
    if image_data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if image_data[:2] == b'\xff\xd8':
        return "image/jpeg"
    if image_data[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    if image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
        return "image/webp"
    if image_data[:2] == b'BM':
        return "image/bmp"
    return "image/png"


# Map MIME type to file extension
MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}
