#!/usr/bin/env python3
"""
ConnectClaw — AI coding agent connected to Feishu IM.

Usage:
  connectclaw              Start the bot
  connectclaw onboard      Run setup wizard
  connectclaw --home DIR   Use custom config directory
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys

from connectclaw.channel.feishu import FeishuChannel  # noqa: E402 — must load before asyncio loop
from connectclaw.logging import get_logger

logger = get_logger(__name__)


def _parse_args(argv: list[str]) -> dict:
    """Parse CLI arguments. Returns {onboard: bool, home: str|None}."""
    result = {"onboard": False, "home": None}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("onboard", "--onboard"):
            result["onboard"] = True
        elif arg == "--home" and i + 1 < len(argv):
            i += 1
            result["home"] = argv[i]
        elif arg.startswith("--home="):
            result["home"] = arg.split("=", 1)[1]
        i += 1
    return result


async def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_args(argv)

    # Set home directory
    if args["home"]:
        os.environ["CONNECTCLAW_HOME"] = os.path.expanduser(args["home"])
        os.environ["CONNECTCLAW_CONFIG"] = os.path.join(
            os.path.expanduser(args["home"]), "config.toml"
        )

    # Route to onboard or bot
    if args["onboard"]:
        from connectclaw.onboard import run_onboard
        await run_onboard()
        return

    # Start bot
    from connectclaw.commands import handle as handle_command
    from connectclaw.config import Config
    from connectclaw.coding.coding_agent import CodingAgent

    config = Config.load()
    logger.info("ConnectClaw starting...")
    logger.info("  LLM: %s", config.llm.model_id)
    logger.info("  Thinking: %s", config.agent.thinking_level)
    logger.info("  CWD: %s", config.agent.cwd)
    logger.info("  Sessions: %s", config.session.dir)
    logger.info("  RAG: %s", "enabled" if config.rag.enabled else "disabled")
    glyph_available = shutil.which(config.web_search.glyph_bin) is not None or os.path.isfile(
        os.path.expanduser("~/.cargo/bin/glyph")
    )
    logger.info("  Web Search: %s", "available (glyph)" if glyph_available else "glyph not found — git clone https://github.com/k1y0miiii/glyph.git && cd glyph && cargo install --locked --path crates/app")
    logger.info("  Vision: %s", "configured" if config.vision.api_key else "not configured")

    if not config.llm.api_key:
        logger.error("LLM API key is required.")
        logger.error("  Run 'connectclaw onboard' to configure,")
        logger.error("  or set LLM_API_KEY in .env")
        sys.exit(1)

    # Create channel first, then agent (agent needs channel for auth cards)
    channel = FeishuChannel(config.feishu)
    coding_agent = CodingAgent(config, channel=channel)

    # Initialize RAG if enabled
    if config.rag.enabled:
        await coding_agent.initialize_rag()

    async def on_message(conversation_key: str, text: str, live_card_callbacks: dict | None = None) -> str | None:
        logger.info("[%s] User: %s", conversation_key[:8], text[:100])

        # Dispatch slash commands
        command_result = await handle_command(
            text,
            conversation_key=conversation_key,
            agent=coding_agent,
        )
        if command_result is not None:
            return command_result

        try:
            response = await coding_agent.handle_message(conversation_key, text, live_card_callbacks)
            if response:
                logger.info("[%s] Assistant: %s", conversation_key[:8], response[:100])
            return response
        except Exception as e:
            logger.error("[%s] Error: %s", conversation_key[:8], e)
            return f"Error: {e}"

    # Handle graceful shutdown on SIGINT / Ctrl+C
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_sigint() -> None:
        logger.info("Received SIGINT, shutting down...")
        stop_event.set()
        channel.close_safe()

    loop.add_signal_handler(signal.SIGINT, _on_sigint)

    try:
        await channel.start(on_message)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
    finally:
        loop.remove_signal_handler(signal.SIGINT)
        await channel.close()


def cli() -> None:
    """CLI entry point for pyproject.toml scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
