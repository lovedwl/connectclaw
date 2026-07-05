"""Abstract Channel interface for IM platform integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any


OnMessageCallback = Callable[..., Awaitable[str | None]]
"""Callback: (conversation_key, text, live_card_callbacks=None, *,
              resources=None, message_id=None) -> response_text"""


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
