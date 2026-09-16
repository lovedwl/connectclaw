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
import contextlib
import os
import shutil
import signal
import sys

from connectclaw.channel.feishu import FeishuChannel  # noqa: E402 — must load before asyncio loop
from connectclaw.coding.tools.attach_image import (
    AttachmentStore,
    DEFAULT_ATTACHMENTS_DIR,
)
from connectclaw.logging import get_logger

logger = get_logger(__name__)


async def _download_feishu_images(
    *,
    channel: FeishuChannel,
    store: AttachmentStore,
    conversation_key: str,
    resources: list,
    message_id: str,
    max_images: int = 5,
) -> list[dict]:
    """Download Feishu images into the attachments dir and register them.

    Files land in ``~/.connectclaw/attachments/<xxh128>.<ext>`` — content-hash
    names dedupe repeats across messages — and each is registered with the
    in-process AttachmentStore so the ``attach_image`` tool can re-attach it on
    a later turn. Returns a list of image_ref content blocks for this turn.
    """
    from connectclaw.coding.tools.attach_image import make_image_ref
    from connectclaw.coding.tools.image_analyze import MIME_TO_EXT, detect_mime_type

    import xxhash

    image_resources = [r for r in resources if r.type == "image"]
    if not image_resources:
        return []

    attachments_dir = DEFAULT_ATTACHMENTS_DIR
    os.makedirs(attachments_dir, exist_ok=True)

    overflow = len(image_resources) - max_images
    to_process = image_resources[:max_images]

    async def download_one(i: int, res) -> dict | None:
        file_key = res.file_key
        if not file_key:
            return None

        logger.info("[%s] downloading image %d/%d: %s",
                     conversation_key[:8], i + 1, len(to_process), file_key[:20])

        try:
            image_data = await channel.download_resource(
                file_key, resource_type="image", message_id=message_id,
            )
        except Exception as e:
            logger.error("[%s] image download failed: %s", conversation_key[:8], e)
            return None

        if image_data is None:
            logger.warning("[%s] image download returned empty: %s",
                           conversation_key[:8], file_key[:20])
            return None

        mime_type = detect_mime_type(image_data)
        ext = MIME_TO_EXT.get(mime_type, ".png")
        image_id = xxhash.xxh128(image_data).hexdigest()
        filepath = os.path.join(attachments_dir, f"{image_id}{ext}")

        # Content-hash name makes this idempotent: re-sent images are no-ops.
        if not os.path.exists(filepath):
            with open(filepath, "wb") as f:
                f.write(image_data)

        item = await store.register(image_id, filepath, mime_type, len(image_data))

        logger.info("[%s] saved image: id=%s (%s, %dKB)",
                     conversation_key[:8], image_id, mime_type, len(image_data) // 1024)

        return make_image_ref(item)

    # Downloads are independent — run them concurrently instead of serially.
    results = await asyncio.gather(
        *(download_one(i, res) for i, res in enumerate(to_process))
    )
    refs = [r for r in results if r is not None]

    if overflow > 0:
        logger.info("[%s] %d image(s) skipped (max %d)",
                     conversation_key[:8], overflow, max_images)

    return refs


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

    # ── Restart event（供 /restart 命令触发进程重启）──
    # 监控协程只负责「打信号 + 让 channel 主循环退出」，绝不自己调
    # sys.exit —— 子任务里的 SystemExit 会被 asyncio 吞成 task 异常，
    # 退出码传不出去，守护进程也就不会拉起。真正的 SystemExit(42)
    # 由根协程在清理完毕后抛出。
    restart_event = asyncio.Event()
    coding_agent._restart_event = restart_event
    restart_requested = False

    async def _restart_monitor():
        nonlocal restart_requested
        try:
            await restart_event.wait()
        except asyncio.CancelledError:
            return
        logger.info("♻️ 重启信号已收到，等待回复发送完毕…")
        # 给飞书一点时间把回复推送给用户
        await asyncio.sleep(2)
        restart_requested = True
        # 只做一件事：让 channel.start() 的 keep-alive 循环退出，
        # 主流程落到 finally 做统一清理。
        channel.close_safe()

    restart_task = asyncio.create_task(_restart_monitor())

    # China-friendly HuggingFace mirror. The embedding model (memory/RAG) is
    # pulled from HuggingFace; without a mirror the first load can hang for a
    # long time trying to reach huggingface.co. Only set when the user hasn't
    # overridden it. Must be set before sentence-transformers is imported
    # (which happens lazily on first embed), so here at startup is fine.
    # After the model is cached, export HF_HUB_OFFLINE=1 to skip update checks.
    if config.memory.enabled or config.rag.enabled:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        logger.info("  HF endpoint: %s", os.environ["HF_ENDPOINT"])

        # Cap ML thread/process fan-out. On a 24-core box, torch's intra-op
        # pool and joblib/loky each default to one worker PER CORE — the embedder
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
        # Download and register Feishu images before agent processing. They are
        # attached to this turn's user message as image_ref blocks (and recorded
        # in the attachments manifest so attach_image can re-attach them later).
        resources = kwargs.get("resources") or []
        message_id = kwargs.get("message_id", "")
        images: list[dict] | None = None

        if resources:
            saved_images = await _download_feishu_images(
                channel=channel,
                store=coding_agent.attachment_store,
                conversation_key=conversation_key,
                resources=resources,
                message_id=message_id,
                max_images=config.agent.max_images,
            )
            if saved_images:
                images = saved_images
                ids = ", ".join(img["id"] for img in saved_images)
                parts = [text] if text else []
                parts.append(
                    f"---\n图片已直接附加到上下文（id: {ids}）。"
                    f"需要再次查看历史图片时可使用 attach_image 工具。"
                )
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
            response = await coding_agent.handle_message(
                conversation_key, text, live_card_callbacks, images=images
            )
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
        restart_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await restart_task
        if config.memory.enabled:
            await coding_agent.memory.close()
        await coding_agent.attachment_store.flush()
        await channel.close()

    # 根协程抛出 → asyncio.run 会把它传播成进程退出码。
    # 守护进程（systemd 等）据此判定「请拉起」。
    if restart_requested:
        logger.info("♻️ 进程退出（code 42），等待守护进程拉起…")
        raise SystemExit(42)


def cli() -> None:
    """CLI entry point for pyproject.toml scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
