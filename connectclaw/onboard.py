"""
Interactive onboarding wizard for ConnectClaw.

Usage: connectclaw onboard

Steps:
  1. Auto-create Feishu bot (scan QR with Feishu/Lark) or manual config
  2. LLM model + API key
  3. Optional: RAG, web search, vision model
  4. Save config to ~/.connectclaw/config.toml

The QR code creates a new bot APP within your existing Feishu organization.
It does NOT create a new Feishu account.
"""

from __future__ import annotations

import asyncio
import os
import sys

import questionary


CONFIG_DIR = os.path.expanduser("~/.connectclaw")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.toml")
BAR = "│"


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


# ── Step 1: Feishu App Setup ──────────────────────────────────


async def _auto_create_feishu_app() -> dict | None:
    """Try auto-creating a Feishu bot app via lark_oapi.register_app().

    Shows QR code for user to scan with Feishu/Lark. Creates a new bot
    app within the user's existing organization.
    """
    import lark_oapi

    qr_shown = False
    last_error = [None]  # mutable container for callback

    def on_qr_code(info):
        nonlocal qr_shown
        qr_shown = True
        import qrcode as qrlib

        # info is a dict from lark_oapi, not an object
        url = info["url"] if isinstance(info, dict) else info.url
        expire = info.get("expire_in", 300) if isinstance(info, dict) else getattr(info, "expire_in", 300)

        print()
        print(f"{BAR}  {_bold('用飞书 / Lark 扫码创建机器人')}")
        print(f"{BAR}")
        qr = qrlib.QRCode()
        qr.add_data(url)
        qr.print_ascii(invert=True)
        print(f"{BAR}")
        print(f"{BAR}  或打开：{url}")
        print(f"{BAR}  {expire} 秒后过期")
        print(f"{BAR}")
        print(f"{BAR}  等待扫码…")

    def on_status_change(info):
        status = info.get("status", "") if isinstance(info, dict) else getattr(info, "status", "")
        if status == "domain_switched":
            print(f"{BAR}  已切换域名，继续…")

    try:
        result = await lark_oapi.aregister_app(
            on_qr_code=on_qr_code,
            on_status_change=on_status_change,
            source="connectclaw",
        )

        # result may be a dict or an object
        if isinstance(result, dict):
            client_id = result.get("client_id", "")
            client_secret = result.get("client_secret", "")
            user_info = result.get("user_info", {})
            tenant_brand = user_info.get("tenant_brand", "") if isinstance(user_info, dict) else ""
        else:
            client_id = getattr(result, "client_id", "")
            client_secret = getattr(result, "client_secret", "")
            ui = getattr(result, "user_info", None)
            tenant_brand = str(getattr(ui, "tenant_brand", "")).lower() if ui else ""

        if client_id and client_secret:
            brand = "lark" if "lark" in str(tenant_brand).lower() else "feishu"
            print(f"\n{BAR}  机器人已创建：{client_id}")
            return {"app_id": client_id, "app_secret": client_secret, "brand": brand}

        print(f"\n{BAR}  自动创建返回的数据不完整，回退到手动配置…")
        return None

    except Exception as e:
        msg = str(e)
        # Common failure modes:
        # - "source not found": need to use an approved source ID
        # - Network/timeout issues
        if not qr_shown:
            print(f"\n{BAR}  自动创建不可用：{msg[:120]}")
        else:
            print(f"\n{BAR}  {msg[:120]}")
        return None


async def _manual_feishu_config() -> dict:
    """Manual Feishu app configuration."""
    print()
    print(f"  {_bold('手动配置应用')}")
    print(f"  前往 https://open.feishu.cn → 开发者后台 → 创建应用")
    print(f"  然后在「凭证与基础信息」中获取 App ID 与 App Secret。")
    print()

    app_id = await questionary.text(
        "App ID：",
        validate=lambda v: "必填" if not v.strip() else True,
    ).ask_async()
    if not app_id:
        sys.exit(0)

    app_secret = await questionary.password(
        "App Secret：",
        validate=lambda v: "必填" if not v.strip() else True,
    ).ask_async()
    if not app_secret:
        sys.exit(0)

    brand = await questionary.select(
        "区域：",
        choices=[
            {"name": "飞书（中国）", "value": "feishu"},
            {"name": "Lark（国际版）", "value": "lark"},
        ],
    ).ask_async()

    return {"app_id": app_id.strip(), "app_secret": app_secret.strip(), "brand": brand or "feishu"}


async def _pick_feishu_setup(existing: dict | None = None) -> dict:
    """Step 1. If existing config found, skip unless user wants to change."""
    if existing:
        keep = await questionary.confirm(
            f"飞书应用：{existing['app_id']} — 是否保留？",
            default=True,
        ).ask_async()
        if keep:
            return existing

    mode = await questionary.select(
        "飞书 / Lark 应用配置：",
        choices=[
            {"name": "创建新机器人（扫码）", "value": "auto"},
            {"name": "使用已有应用（粘贴凭证）", "value": "manual"},
        ],
    ).ask_async()

    if mode == "auto":
        result = await _auto_create_feishu_app()
        if result:
            return result
        print(f"\n{BAR}  回退到手动配置…")
        return await _manual_feishu_config()
    return await _manual_feishu_config()


# ── Step 2: Model ─────────────────────────────────────────────


async def _pick_model(existing: dict | None = None) -> dict:
    """Step 2."""
    if existing:
        print()
        print(f"  {_bold('大模型')}（已保存：{existing.get('model_id', '?')}）")
        keep = await questionary.confirm("是否保留现有模型配置？", default=True).ask_async()
        if keep:
            return existing

    print()
    print(f"  {_bold('大模型')}")

    default_model = existing.get("model_id", "deepseek-chat") if existing else "deepseek-chat"
    model_id = await questionary.text("模型 ID：", default=default_model).ask_async() or default_model

    env_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    saved_key = existing.get("api_key", "") if existing else ""
    default_key = env_key or saved_key
    hint = "（已找到）" if default_key else ""

    api_key = await questionary.password(f"大模型 API Key{hint}：").ask_async()
    key = (api_key or "").strip() or default_key

    if not key:
        print("  警告：未设置 API Key。")

    return {"model_id": model_id.strip(), "api_key": key}


# ── Step 3: Options ───────────────────────────────────────────


async def _pick_options(existing: dict | None = None) -> dict:
    """Step 3."""
    if existing:
        print()
        print(f"  {_bold('可选功能')}（已保存）")
        keep = await questionary.confirm("是否保留现有设置？", default=True).ask_async()
        if keep:
            return existing

    print()
    print(f"  {_bold('可选功能')}")

    default_thinking = existing.get("thinking_level", "off") if existing else "off"
    thinking = await questionary.select(
        "思考等级：",
        choices=[
            {"name": "关闭 (off)", "value": "off"},
            {"name": "最少 (minimal)", "value": "minimal"},
            {"name": "低 (low)", "value": "low"},
            {"name": "中 (medium)", "value": "medium"},
            {"name": "高 (high)", "value": "high"},
        ],
        default=default_thinking,
    ).ask_async()

    rag = await questionary.confirm("启用 RAG（文档检索，约需下载 2GB 模型）？", default=False).ask_async()
    docs = ""
    if rag:
        docs = await questionary.path("文档目录：", default=".", only_directories=True).ask_async() or ""

    vision_api_key = await questionary.password("视觉模型 API Key（可选）：").ask_async()
    vision_model_id = ""
    vision_base_url = ""
    if vision_api_key and vision_api_key.strip():
        vision_model_id = await questionary.text("视觉模型 ID：", default="qwen3-vl-plus").ask_async() or ""
        vision_base_url = await questionary.text("视觉模型 Base URL：", default="https://dashscope.aliyuncs.com/compatible-mode/v1").ask_async() or ""

    return {
        "thinking_level": thinking or "off",
        "rag_enabled": bool(rag and docs),
        "rag_docs_dir": docs or "",
        "vision_api_key": (vision_api_key or "").strip(),
        "vision_model_id": vision_model_id.strip(),
        "vision_base_url": vision_base_url.strip(),
    }


# ── Write Config ──────────────────────────────────────────────


def _write_config(feishu: dict, model: dict, opts: dict) -> str:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    c = (
        f"# ConnectClaw 配置\n\n"
        f"[llm]\napi_key = \"{model['api_key']}\"\nbase_url = \"https://api.deepseek.com\"\nmodel_id = \"{model['model_id']}\"\n\n"
        f"[feishu]\napp_id = \"{feishu['app_id']}\"\napp_secret = \"{feishu['app_secret']}\"\n\n"
        f"[vision]\napi_key = \"{opts['vision_api_key']}\"\nbase_url = \"{opts['vision_base_url']}\"\nmodel_id = \"{opts['vision_model_id']}\"\n\n"
        f"[agent]\ncwd = \"{os.getcwd()}\"\nthinking_level = \"{opts['thinking_level']}\"\n\n"
        f"[session]\ndir = \"~/.connectclaw/sessions\"\n\n"
        f"[rag]\nenabled = {str(opts['rag_enabled']).lower()}\ndocs_dir = \"{opts['rag_docs_dir']}\"\ndb_path = \"~/.connectclaw/rag_db\"\ntop_k = 20\ntop_n = 5\n\n"
        f"[web_search]\nmax_chars = 8000\ntimeout = 30\n\n"
        f"[compaction]\nenabled = true\nreserve_tokens = 16384\nkeep_recent_tokens = 20000\n"
    )
    with open(CONFIG_PATH, "w") as f:
        f.write(c)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass
    return CONFIG_PATH


# ── Config Loading ──────────────────────────────────────────


def _load_existing_config() -> dict | None:
    """Load existing config if present, return normalized dict or None."""
    if not os.path.exists(CONFIG_PATH):
        return None

    from connectclaw.config import Config
    try:
        c = Config.load(CONFIG_PATH)
    except Exception:
        return None

    feishu = None
    if c.feishu.app_id and c.feishu.app_secret:
        feishu = {"app_id": c.feishu.app_id, "app_secret": c.feishu.app_secret, "brand": "feishu"}

    model = None
    if c.llm.api_key:
        model = {"model_id": c.llm.model_id, "api_key": c.llm.api_key}

    opts = {
        "thinking_level": c.agent.thinking_level,
        "rag_enabled": c.rag.enabled,
        "rag_docs_dir": c.rag.docs_dir,
        "vision_api_key": c.vision.api_key,
        "vision_model_id": c.vision.model_id,
        "vision_base_url": c.vision.base_url,
    }

    return {"feishu": feishu, "model": model, "opts": opts}


# ── Main ──────────────────────────────────────────────────────


async def run_onboard() -> None:
    # Load existing config if present (never None so later .get() is safe)
    existing = _load_existing_config() or {}

    if existing:
        print()
        print(f"{BAR}  {_bold('ConnectClaw — 更新配置')}")
        print(f"{BAR}")
        if existing.get("feishu"):
            f = existing["feishu"]
            print(f"{BAR}  飞书：{f['app_id']}（已保存）")
        if existing.get("model"):
            m = existing["model"]
            print(f"{BAR}  模型：{m['model_id']}（已保存）")
        print(f"{BAR}")
        print(f"{BAR}  直接回车保留现有值，或输入新值。")
        print(f"{BAR}")
    else:
        print()
        print(f"{BAR}  {_bold('ConnectClaw 安装向导')}")
        print(f"{BAR}")
        print(f"{BAR}  1. 创建飞书/Lark 机器人（扫码）")
        print(f"{BAR}  2. 大模型 API Key 与模型")
        print(f"{BAR}  3. 可选：RAG、网页搜索、视觉模型")
        print(f"{BAR}")

    feishu = await _pick_feishu_setup(existing.get("feishu"))
    model = await _pick_model(existing.get("model"))
    opts = await _pick_options(existing.get("opts"))

    path = _write_config(feishu, model, opts)
    print(f"\n  配置文件：{path}")

    start = await questionary.confirm("现在启动？", default=True).ask_async()
    if start:
        print("\n  正在启动 ConnectClaw…\n")
        from connectclaw.main import main
        await main([])  # empty args → skip onboard, start bot
    else:
        print(f"\n{BAR}  运行：connectclaw")
        print(f"{BAR}  配置文件：{path}\n")
