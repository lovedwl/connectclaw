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


async def _download_feishu_images(
    *,
    channel: FeishuChannel,
    conversation_key: str,
    resources: list,
    message_id: str,
    max_images: int = 5,
) -> list[dict]:
    """Download Feishu images to local cache.

    Saves images to ``~/.cache/cc/i/`` with short sequential names
    (1.png, 2.jpg, ...).  Returns a list of dicts with path, mime_type,
    and size_bytes for each successfully downloaded image.
    """
    from connectclaw.coding.tools.image_analyze import MIME_TO_EXT, detect_mime_type

    image_resources = [r for r in resources if r.type == "image"]
    if not image_resources:
        return []

    cache_dir = os.path.expanduser("~/.cache/cc/i")
    os.makedirs(cache_dir, exist_ok=True)

    overflow = len(image_resources) - max_images
    to_process = image_resources[:max_images]

    saved = []
    for i, res in enumerate(to_process):
        file_key = res.file_key
        if not file_key:
            continue

        logger.info("[%s] downloading image %d/%d: %s",
                     conversation_key[:8], i + 1, len(to_process), file_key[:20])

        try:
            image_data = await channel.download_resource(
                file_key, resource_type="image", message_id=message_id,
            )
        except Exception as e:
            logger.error("[%s] image download failed: %s", conversation_key[:8], e)
            continue

        if image_data is None:
            logger.warning("[%s] image download returned empty: %s",
                           conversation_key[:8], file_key[:20])
            continue

        mime_type = detect_mime_type(image_data)
        ext = MIME_TO_EXT.get(mime_type, ".png")
        filename = f"{len(saved) + 1}{ext}"
        filepath = os.path.join(cache_dir, filename)

        with open(filepath, "wb") as f:
            f.write(image_data)

        size_kb = len(image_data) // 1024
        logger.info("[%s] saved image: %s (%s, %dKB)",
                     conversation_key[:8], filepath, mime_type, size_kb)

        saved.append({
            "path": f"~/.cache/cc/i/{filename}",
            "mime_type": mime_type,
            "size_bytes": len(image_data),
        })

    if overflow > 0:
        logger.info("[%s] %d image(s) skipped (max %d)",
                     conversation_key[:8], overflow, max_images)

    return saved


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
    logger.info("  Memory: %s", "enabled" if config.memory.enabled else "disabled")
    try:
        import lightpanda  # noqa: F401
        browser_ok = True
    except Exception:
        browser_ok = False
    logger.info("  Web/Browser: %s", "Lightpanda ready" if browser_ok else "lightpanda-py missing — uv add lightpanda-py")
    logger.info("  Vision: %s", "configured" if config.vision.api_key else "not configured")

    if not config.llm.api_key:
        logger.error("LLM API key is required.")
        logger.error("  Run 'connectclaw onboard' to configure,")
        logger.error("  or set LLM_API_KEY in .env")
        sys.exit(1)

    # Create channel first, then agent (agent needs channel for auth cards)
    channel = FeishuChannel(config.feishu)
    coding_agent = CodingAgent(config, channel=channel)

    # China-friendly HuggingFace mirror. BGE-M3 (memory/RAG embeddings) is
    # pulled from HuggingFace; without a mirror the first load can hang for a
    # long time trying to reach huggingface.co. Only set when the user hasn't
    # overridden it. Must be set before sentence-transformers is imported
    # (which happens lazily on first embed), so here at startup is fine.
    # After the model is cached, export HF_HUB_OFFLINE=1 to skip update checks.
    if config.memory.enabled or config.rag.enabled:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        logger.info("  HF endpoint: %s", os.environ["HF_ENDPOINT"])

        # Cap ML thread/process fan-out. On a 24-core box, torch's intra-op
        # pool and joblib/loky each default to one worker PER CORE — BGE-M3
        # runs on the per-turn recall path, so an unbounded fan-out spikes CPU
        # and RSS (and leaks loky semaphores at shutdown). A small fixed cap is
        # plenty for single-query embedding and keeps the baseline flat. Must be
        # set before torch / sentence-transformers import (lazy, on first embed).
        _ml_threads = os.environ.setdefault("CONNECTCLAW_ML_THREADS", "4")
        os.environ.setdefault("OMP_NUM_THREADS", _ml_threads)
        os.environ.setdefault("MKL_NUM_THREADS", _ml_threads)
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", _ml_threads)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        logger.info("  ML threads capped at %s", _ml_threads)

    # Initialize RAG if enabled
    if config.rag.enabled:
        await coding_agent.initialize_rag()

    # Initialize memory if enabled
    if config.memory.enabled:
        await coding_agent.memory.initialize()
        await coding_agent.memory.schedule_dreaming(
            coding_agent._model,
            api_key=config.llm.api_key or None,
        )

    async def on_message(
        conversation_key: str,
        text: str,
        live_card_callbacks: dict | None = None,
        **kwargs,
    ) -> str | None:
        # Download and cache Feishu images before agent processing
        resources = kwargs.get("resources") or []
        message_id = kwargs.get("message_id", "")

        if resources:
            saved_images = await _download_feishu_images(
                channel=channel,
                conversation_key=conversation_key,
                resources=resources,
                message_id=message_id,
                max_images=config.agent.max_images,
            )
            if saved_images:
                parts = [text] if text else []
                parts.append("---")
                if len(saved_images) == 1:
                    parts.append("用户发送了 1 张图片，可使用 image_analyze 工具查看：")
                else:
                    parts.append(f"用户发送了 {len(saved_images)} 张图片，可使用 image_analyze 工具查看：")
                for img in saved_images:
                    size_kb = img["size_bytes"] // 1024
                    parts.append(f"- {img['path']} ({img['mime_type']}, {size_kb}KB)")
                text = "\n".join(parts)

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
        if config.memory.enabled:
            await coding_agent.memory.close()
        await channel.close()


def cli() -> None:
    """CLI entry point for pyproject.toml scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
