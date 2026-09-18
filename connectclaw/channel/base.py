"""Abstract Channel interface for IM platform integration."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any


OnMessageCallback = Callable[..., Awaitable[str | None]]
"""Callback: (conversation_key, text, live_card_callbacks=None, *,
              resources=None, message_id=None, sender_open_id=None)
         -> response_text"""


MEDIA_PLACEHOLDER_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
"""The SDK flattens inbound image/sticker messages into markdown placeholders
(``![image](img_v3_...)``)."""


def is_media_placeholder(text: str) -> bool:
    """True if ``text`` starts with a media placeholder like ``![image](key)``.

    Such messages start with ``!`` but are inbound media, never an operator
    ``!`` command — both the channel live-card gate and the operator-bash
    gate must exclude them.
    """
    return bool(MEDIA_PLACEHOLDER_RE.match(text))


class Channel(ABC):
    """Abstract interface for IM platform integration."""

    @abstractmethod
    async def start(self, on_message: OnMessageCallback) -> None:
        """
        Start listening for messages. Blocks until stopped.

        Args:
            on_message: Called when a message arrives.
                Args: (conversation_key, text)
                Returns: response text to send back
        """
        ...

    @abstractmethod
    async def send_message(self, conversation_key: str, text: str) -> str:
        """Send a text message. Returns message_id."""
        ...

    @abstractmethod
    async def send_card(self, conversation_key: str, card: dict[str, Any]) -> str:
        """Send an interactive card. Returns message_id."""
        ...

    # ── Outbound media capabilities ─────────────────────────
    # Channel-owned capabilities (files, images, cards, live streaming, ...).
    # The agent binds to these through the abstract interface only — swap the
    # channel implementation or add new capabilities here, never in the agent
    # layer (see channel/capabilities.py).

    @abstractmethod
    async def send_image(self, conversation_key: str, image_path: str) -> str:
        """Upload a local image and send it as an image message. Returns
        message_id, or "" on failure."""
        ...

    @abstractmethod
    async def send_file(self, conversation_key: str, file_path: str) -> str:
        """Upload a local file and send it as a file attachment. Returns
        message_id, or "" on failure."""
        ...

    @abstractmethod
    async def send_thinking_indicator(self, conversation_key: str) -> None:
        """Show a typing/thinking indicator."""
        ...

    @abstractmethod
    async def send_error(self, conversation_key: str, error: str) -> str:
        """Send an error message."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Gracefully close the channel."""
        ...
