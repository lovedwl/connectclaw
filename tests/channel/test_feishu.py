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


def test_only_stop_bypasses_serialization():
    """/stop 是唯一必须绕过串行锁的命令（否则会死在它要取消的任务后面）。"""
    ch = _channel()
    assert ch._is_interrupt("/stop") is True
    assert ch._is_interrupt("  /stop") is True      # 带前导空格
    assert ch._is_interrupt("/stop 额外参数") is True
    assert ch._is_interrupt("/forget 1") is False   # 走正常串行
    assert ch._is_interrupt("普通消息") is False


def test_media_placeholder_is_not_a_command():
    """SDK 把入站图片渲染成 `![image](key)` —— 以 ! 开头但不是命令。

    图片消息是正常的 agent 回合（图片会被下载并附加到上下文），既不能进
    operator bash 分支，也不能被跳过 live 思考卡。
    """
    ch = _channel()
    img = "![image](img_v3_0215k_fb65a176-562a-4014-8d2c-71273f0ecbag)"
    assert ch._is_cmd(img) is False
    assert ch._is_cmd("![表情](img_v3_xxx) 后面带文字") is False
    assert ch._is_cmd("!pwd") is True
    assert ch._is_cmd("/model list") is True


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


# ── live card gate (wizard replies are not agent turns) ────────


def test_card_suppressed_for_commands_and_wizard_replies():
    """向导回复不该弹「思考中」卡片：卡片是立刻发的，没人会去更新它。"""
    ch = _channel()
    # 普通消息 → 有卡片
    assert ch._suppress_card("oc", "你好") is False
    # 命令 / operator escape → 无卡片（原有行为）
    assert ch._suppress_card("oc", "/model add") is True
    assert ch._suppress_card("oc", "!ls") is True
    # 装上网关后，向导等待中的回复也走无卡片路径
    ch.set_card_gate(lambda chat_id, text: chat_id == "oc_wizard")
    assert ch._suppress_card("oc_wizard", "https://gw/v1") is True
    assert ch._suppress_card("oc", "https://gw/v1") is False


def test_card_gate_failure_never_breaks_message_handling():
    ch = _channel()

    def boom(chat_id, text):
        raise RuntimeError("gate exploded")

    ch.set_card_gate(boom)
    assert ch._suppress_card("oc", "你好") is False
