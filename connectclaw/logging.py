"""
Unified logging for ConnectClaw.

Usage:
    from connectclaw.logging import get_logger
    logger = get_logger(__name__)
    logger.debug("...")
    logger.info("...")
    logger.warning("...")
    logger.error("...")

Level: controlled by CONNECTCLAW_LOG_LEVEL env var (default: INFO).
"""

from __future__ import annotations

import logging
import os
import sys

_logging_configured = False

# Format: "HH:MM:SS.mmm [LEVEL  ] [name          ] message"
_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-7s] [%(name)-14s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class _NameShortener(logging.Filter):
    """Shorten logger names for display: 'connectclaw.a.b.c' → 'a.b.c'.

    Keeps the last 2 segments for deeply nested names
    (e.g. 'connectclaw.agent.harness.rag.subsystem' → 'rag.subsystem').
    """

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if name.startswith("connectclaw."):
            name = name[len("connectclaw."):]
        parts = name.split(".")
        if len(parts) > 2:
            name = ".".join(parts[-2:])
        record.name = name
        return True


def setup_logging(level: str | None = None) -> None:
    """Configure root logger once. Idempotent."""
    global _logging_configured
    if _logging_configured:
        return

    if level is None:
        level = os.environ.get("CONNECTCLAW_LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_NameShortener())

    root = logging.getLogger("connectclaw")
    root.setLevel(getattr(logging, level, logging.INFO))
    root.addHandler(handler)
    root.propagate = False

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the 'connectclaw' namespace.

    Always returns a child of the 'connectclaw' logger so the handler
    and formatter are inherited.  The display name is shortened by
    _NameShortener.
    """
    setup_logging()
    return logging.getLogger(name)
