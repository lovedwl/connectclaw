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
        print(f"{BAR}  {_bold('Scan with Feishu / Lark to create bot')}")
        print(f"{BAR}")
        qr = qrlib.QRCode()
        qr.add_data(url)
        qr.print_ascii(invert=True)
        print(f"{BAR}")
        print(f"{BAR}  Or open: {url}")
        print(f"{BAR}  Expires in {expire}s")
        print(f"{BAR}")
        print(f"{BAR}  Waiting for scan...")

    def on_status_change(info):
        status = info.get("status", "") if isinstance(info, dict) else getattr(info, "status", "")
        if status == "domain_switched":
            print(f"{BAR}  Switched domain, continuing...")

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
            print(f"\n{BAR}  Bot created: {client_id}")
            return {"app_id": client_id, "app_secret": client_secret, "brand": brand}

        print(f"\n{BAR}  Auto-creation returned incomplete data, falling back...")
        return None

    except Exception as e:
        msg = str(e)
        # Common failure modes:
        # - "source not found": need to use an approved source ID
        # - Network/timeout issues
        if not qr_shown:
            print(f"\n{BAR}  Auto-creation unavailable: {msg[:120]}")
        else:
            print(f"\n{BAR}  {msg[:120]}")
        return None


async def _manual_feishu_config() -> dict:
    """Manual Feishu app configuration."""
    print()
    print(f"  {_bold('Manual App Setup')}")
    print(f"  Go to https://open.feishu.cn → Developer Console → Create App")
    print(f"  Then get App ID & App Secret from Credentials & Basic Info.")
    print()

    app_id = await questionary.text(
        "App ID:",
        validate=lambda v: "Required" if not v.strip() else True,
    ).ask_async()
    if not app_id:
        sys.exit(0)

    app_secret = await questionary.password(
        "App Secret:",
        validate=lambda v: "Required" if not v.strip() else True,
    ).ask_async()
    if not app_secret:
        sys.exit(0)

    brand = await questionary.select(
        "Region:",
        choices=[
            {"name": "Feishu (China)", "value": "feishu"},
            {"name": "Lark (International)", "value": "lark"},
        ],
    ).ask_async()

    return {"app_id": app_id.strip(), "app_secret": app_secret.strip(), "brand": brand or "feishu"}


async def _pick_feishu_setup(existing: dict | None = None) -> dict:
    """Step 1. If existing config found, skip unless user wants to change."""
    if existing:
        keep = await questionary.confirm(
            f"Feishu app: {existing['app_id']} — keep?",
            default=True,
        ).ask_async()
        if keep:
            return existing

    mode = await questionary.select(
        "Feishu / Lark app setup:",
        choices=[
            {"name": "Create new bot (scan QR code)", "value": "auto"},
            {"name": "Use existing app (paste credentials)", "value": "manual"},
        ],
    ).ask_async()

    if mode == "auto":
        result = await _auto_create_feishu_app()
        if result:
            return result
        print(f"\n{BAR}  Falling back to manual setup...")
        return await _manual_feishu_config()
    return await _manual_feishu_config()


# ── Step 2: Model ─────────────────────────────────────────────


async def _pick_model(existing: dict | None = None) -> dict:
    """Step 2."""
    if existing:
        print()
        print(f"  {_bold('LLM Model')} (saved: {existing.get('model_id', '?')})")
        keep = await questionary.confirm("Keep existing model config?", default=True).ask_async()
        if keep:
            return existing

    print()
    print(f"  {_bold('LLM Model')}")

    default_model = existing.get("model_id", "deepseek-chat") if existing else "deepseek-chat"
    model_id = await questionary.text("Model ID:", default=default_model).ask_async() or default_model

    env_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    saved_key = existing.get("api_key", "") if existing else ""
    default_key = env_key or saved_key
    hint = f" (found)" if default_key else ""

    api_key = await questionary.password(f"LLM API Key{hint}:").ask_async()
    key = (api_key or "").strip() or default_key

    if not key:
        print("  Warning: No API key set.")

    return {"model_id": model_id.strip(), "api_key": key}


# ── Step 3: Options ───────────────────────────────────────────


async def _pick_options(existing: dict | None = None) -> dict:
    """Step 3."""
    if existing:
        print()
        print(f"  {_bold('Optional Features')} (saved)")
        keep = await questionary.confirm("Keep existing settings?", default=True).ask_async()
        if keep:
            return existing

    print()
    print(f"  {_bold('Optional Features')}")

    default_thinking = existing.get("thinking_level", "off") if existing else "off"
    thinking = await questionary.select(
        "Thinking level:",
        choices=["off", "minimal", "low", "medium", "high"],
        default=default_thinking,
    ).ask_async()

    rag = await questionary.confirm("Enable RAG (document search, ~2GB model download)?", default=False).ask_async()
    docs = ""
    if rag:
        docs = await questionary.path("Document directory:", default=".", only_directories=True).ask_async() or ""

    vision_api_key = await questionary.password("Vision Model API Key (optional):").ask_async()
    vision_model_id = ""
    vision_base_url = ""
    if vision_api_key and vision_api_key.strip():
        vision_model_id = await questionary.text("Vision Model ID:", default="qwen3-vl-plus").ask_async() or ""
        vision_base_url = await questionary.text("Vision Base URL:", default="https://dashscope.aliyuncs.com/compatible-mode/v1").ask_async() or ""

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
        f"# ConnectClaw Configuration\n\n"
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
        print(f"{BAR}  {_bold('ConnectClaw — Update Config')}")
        print(f"{BAR}")
        if existing.get("feishu"):
            f = existing["feishu"]
            print(f"{BAR}  Feishu: {f['app_id']} (saved)")
        if existing.get("model"):
            m = existing["model"]
            print(f"{BAR}  Model:  {m['model_id']} (saved)")
        print(f"{BAR}")
        print(f"{BAR}  Press Enter to keep existing values, or type new ones.")
        print(f"{BAR}")
    else:
        print()
        print(f"{BAR}  {_bold('ConnectClaw Setup Wizard')}")
        print(f"{BAR}")
        print(f"{BAR}  1. Create Feishu/Lark bot (scan QR)")
        print(f"{BAR}  2. LLM API key + model")
        print(f"{BAR}  3. Optional: RAG, web search, vision model")
        print(f"{BAR}")

    feishu = await _pick_feishu_setup(existing.get("feishu"))
    model = await _pick_model(existing.get("model"))
    opts = await _pick_options(existing.get("opts"))

    path = _write_config(feishu, model, opts)
    print(f"\n  Config: {path}")

    start = await questionary.confirm("Start now?", default=True).ask_async()
    if start:
        print("\n  Starting ConnectClaw...\n")
        from connectclaw.main import main
        await main([])  # empty args → skip onboard, start bot
    else:
        print(f"\n{BAR}  Run: connectclaw")
        print(f"{BAR}  Config: {path}\n")
