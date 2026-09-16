"""Feishu channel per-conversation serialization tests.

Same-chat messages must process in arrival order (a /forget must land before a
follow-up message reads the session); different chats must stay concurrent.
"""

from __future__ import annotations

import asyncio

from connectclaw.channel.feishu import FeishuChannel
from connectclaw.config import FeishuConfig


def _channel() -> FeishuChannel:
    return FeishuChannel(FeishuConfig(app_id="app", app_secret="secret"))


def test_same_chat_gets_one_shared_lock():
    ch = _channel()
    assert ch._chat_lock("oc_a") is ch._chat_lock("oc_a")


def test_different_chats_get_distinct_locks():
    ch = _channel()
    assert ch._chat_lock("oc_a") is not ch._chat_lock("oc_b")


async def test_same_chat_messages_run_serially():
    ch = _channel()
    log: list[str] = []

    async def run(name: str, delay: float) -> None:
        async with ch._chat_lock("oc_shared"):
            log.append(f"{name}:in")
            await asyncio.sleep(delay)
            log.append(f"{name}:out")

    # second 的 delay 远小于 first：若没有锁，second 会插队；有锁则必须等 first 完。
    await asyncio.gather(
        run("first", 0.05),
        run("second", 0.001),
    )
    assert log == ["first:in", "first:out", "second:in", "second:out"]


async def test_different_chats_stay_concurrent():
    ch = _channel()
    log: list[str] = []

    async def run(chat: str, name: str, delay: float) -> None:
        async with ch._chat_lock(chat):
            log.append(f"{name}:in")
            await asyncio.sleep(delay)
            log.append(f"{name}:out")

    await asyncio.gather(run("oc_a", "a", 0.05), run("oc_b", "b", 0.001))
    # 两个会话并行：b 的 in/out 都插在 a 的 out 之前。
    assert log.index("a:in") < log.index("b:in") < log.index("b:out") < log.index("a:out")
