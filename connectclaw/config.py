"""
Configuration management via TOML file with env var override.

Priority: env var > config.toml > config.toml.template defaults

Loads config.toml from:
  1. CONNECTCLAW_CONFIG env var path
  2. ./config.toml (current directory)
  3. ~/.connectclaw/config.toml
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from connectclaw.logging import get_logger

logger = get_logger(__name__)


def _expand_env(value: str) -> str:
    """Expand ${VAR} or $VAR patterns in a string value from environment."""
    pattern = re.compile(r'\$\{(\w+)\}|\$(\w+)')
    def replace(match):
        var_name = match.group(1) or match.group(2)
        val = os.environ.get(var_name)
        if val is None:
            # M42：缺失变量静默变空串会误导排障——至少告警一次
            logger.warning("Environment variable %r referenced in config is not set; expanding to empty string", var_name)
            return ""
        return val
    return pattern.sub(replace, value)


def _safe_int(value: Any, default: int) -> int:
    """M36：宽容的数值转换——支持 '64k'/'8m' 后缀，非法值回退默认并告警。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower().replace("_", "")
    try:
        if s.endswith("k"):
            return int(float(s[:-1]) * 1024)
        if s.endswith("m"):
            return int(float(s[:-1]) * 1024 * 1024)
        return int(s)
    except (ValueError, TypeError):
        logger.warning("Invalid numeric config value %r; using default %d", value, default)
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    """M36：统一 bool 归一——接受 1/true/yes/on 与 0/false/no/off。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    logger.warning("Invalid bool config value %r; using default %s", value, default)
    return default


def _safe_float(value: Any, default: float) -> float:
    """M36：宽容的浮点转换（显式区分 0 与缺失，不像 `or` 会吞掉 0）。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid float config value %r; using default %s", value, default)
        return default


def _load_toml(path: str) -> dict[str, Any]:
    """Load a TOML file. Returns empty dict if not found."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.error("tomli/tomllib not available")
            return {}

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        # M43：TOML 语法错误不再吞掉照常启动——既没 key 又连错端点的
        # 静默降级比启动失败更难排查。
        logger.error("Failed to load %s: %s", path, e)
        raise


def _find_config_path() -> str | None:
    """Find the config.toml file."""
    # 1. Env var
    env_path = os.environ.get("CONNECTCLAW_CONFIG")
    if env_path:
        expanded = os.path.expanduser(env_path)
        if os.path.isfile(expanded):
            return expanded

    # 2. Current directory
    cwd_path = os.path.join(os.getcwd(), "config.toml")
    if os.path.isfile(cwd_path):
        return cwd_path

    # 3. User config directory
    home_path = os.path.expanduser("~/.connectclaw/config.toml")
    if os.path.isfile(home_path):
        return home_path

    return None


# ── Configuration Dataclasses ──────────────────────────────────


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model_id: str = "deepseek-chat"
    context_window: int = 65536
    max_tokens: int = 8192
    reasoning: bool = True


@dataclass
class ProxyConfig:
    """Outbound HTTP(S) proxy — the single config place.

    Currently consumed by the model clients (LLM + retained vision only);
    wire additional outbound consumers (feishu SDK, future outbound APIs) here as
    they need it. Env override: CONNECTCLAW_PROXY_URL.
    """

    url: str = ""


@dataclass
class FeishuConfig:
    app_id: str = ""
    app_secret: str = ""


@dataclass
class VisionConfig:
    api_key: str = ""
    base_url: str = ""
    model_id: str = ""
    # Vision requests ride the same model proxy as the LLM (falls back to [proxy].url).
    proxy: str = ""


@dataclass
class AgentConfig:
    cwd: str = field(default_factory=os.getcwd)
    thinking_level: str = "off"
    max_images: int = 5
    # Which base primitives are exposed DIRECTLY to the main agent (a whitelist
    # from the tool registry). Named agents + scripted tools are NOT here — they
    # are reached through the single `agents` meta-tool. Default = all base
    # primitives (current behaviour). Unknown names are dropped with a warning;
    # an empty resolved set falls back to ['read','bash'].
    tools: list[str] = field(
        default_factory=lambda: [
            "read", "write", "hash_read", "hash_edit",
            "bash", "web_search", "web_fetch", "attach_image", "view_image",
            "skills", "send_file",
        ]
    )


@dataclass
class BashConfig:
    """Operator gate for the `!`/`/bash` escape hatch (direct host bash, unsandboxed).

    Non-empty allow-list of Feishu open_ids that may use it; EMPTY (default)
    disables the escape entirely. This whitelist (and the model registry) are
    agent-protected: connectclaw/security.py refuses the agent from editing
    their source files.
    """

    operator_open_ids: list[str] = field(default_factory=list)


@dataclass
class SandboxConfig:
    """Which isolation the agent's own bash tool runs under.

    mode:
      "auto"    — detect at runtime (bwrap → unshare → direct). Default.
      "bwrap"   — force bubblewrap; falls back to direct if unusable.
      "unshare" — force plain namespaces; falls back to direct if unusable.
      "none"    — NO isolation: the agent's bash runs directly on the host as
                  the bot user ("完全权限"/jailbreak-the-agent switch).

    ``none`` is a deliberate operator decision to trust the agent completely.
    Consequence: bash can then edit config.toml / models.toml — the files that
    gate the `!` escape hatch and the model registry — so the agent could
    widen its own permissions. write/hash_edit still refuse those paths; bash
    bypasses that refusal. Only set this on a machine you fully trust the
    agent on. This file is agent-read-only, so only the operator can flip it.
    """

    mode: str = "auto"


def normalize_sandbox_mode(mode: str) -> str:
    """Map user spellings onto the four canonical modes (unknown → auto)."""
    m = (mode or "auto").strip().lower()
    aliases = {
        "": "auto", "auto": "auto", "detect": "auto", "default": "auto",
        "bwrap": "bwrap", "bubblewrap": "bwrap",
        "unshare": "unshare", "namespace": "unshare",
        "none": "none", "off": "none", "false": "none", "no": "none",
        "direct": "none", "unsandboxed": "none", "full": "none",
    }
    return aliases.get(m, "auto")


@dataclass
class SessionConfig:
    dir: str = "~/.connectclaw/sessions"


@dataclass
class SkillsConfig:
    """Where the `skills` tool scans for SKILL.md packages.

    Defaults to the machine-wide ZCode skill library (~/.agents/skills); point
    elsewhere (or add more dirs later) via config or CONNECTCLAW_SKILLS_DIR.
    """

    dir: str = "~/.agents/skills"


@dataclass
class RAGConfig:
    enabled: bool = False
    docs_dir: str = ""
    db_path: str = "~/.connectclaw/rag_db"
    top_k: int = 20
    top_n: int = 5


@dataclass
class WebSearchConfig:
    max_chars: int = 8000
    timeout: int = 30
    # Max concurrent browser sessions for web_search/web_fetch. One shared
    # Lightpanda `serve` process hosts them all (multi-client model), so this is
    # a session cap, not a process count — no per-session subprocess.
    pool_size: int = 16
    # models.toml profile name for web_fetch `prompt` extraction (a cheap small
    # model). Empty = follow the active main model (hot-switches with /model).
    extract_model: str = ""


@dataclass
class CompactionConfig:
    enabled: bool = True
    reserve_tokens: int = 16384
    # Compact at this fraction of the usable budget (window - reserve).
    trigger_ratio: float = 0.8
    keep_recent_tokens: int = 20000
    # 压缩产物也会留在上下文里：摘要/记忆快照不封顶 → 压缩后仍超阈值 → 抖动。
    summary_max_tokens: int = 4096
    memory_snapshot_max_tokens: int = 2000


@dataclass
class MemoryConfig:
    enabled: bool = True
    db_path: str = "~/.connectclaw/memory.db"
    extract_after_turn: bool = True
    extract_min_turns: int = 3
    extract_interval_turns: int = 5
    max_context_tokens: int = 2000
    recency_threshold_days: int = 7
    use_embeddings: bool = True
    dream_interval_hours: float = 24.0
    decay_halflife_days: float = 30.0
    consolidation_enabled: bool = True


@dataclass
class Config:
    """Top-level ConnectClaw configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    feishu: FeishuConfig = field(default_factory=FeishuConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    bash: BashConfig = field(default_factory=BashConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def load(cls, path: str | None = None) -> Config:
        """Load configuration from TOML file with env var overrides."""
        config_path = path or _find_config_path()

        raw: dict[str, Any] = {}
        if config_path:
            raw = _load_toml(config_path)

        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict) -> Config:
        """Build Config from raw TOML dict with env var fallback."""

        # LLM (supports [llm] with fallback to legacy [deepseek] section)
        llm_raw = raw.get("llm") or raw.get("deepseek", {})
        llm = LLMConfig(
            api_key=_expand_env(
                os.environ.get("LLM_API_KEY", "")
                or os.environ.get("DEEPSEEK_API_KEY", "")
                or llm_raw.get("api_key", "")
            ),
            base_url=os.environ.get("LLM_BASE_URL")
                or os.environ.get("DEEPSEEK_BASE_URL")
                or llm_raw.get("base_url", "https://api.deepseek.com"),
            model_id=os.environ.get("LLM_MODEL")
                or os.environ.get("DEEPSEEK_MODEL")
                or llm_raw.get("model_id", "deepseek-chat"),
            context_window=_safe_int(llm_raw.get("context_window"), 65536),
            max_tokens=_safe_int(llm_raw.get("max_tokens"), 8192),
            reasoning=_safe_bool(llm_raw.get("reasoning"), True),
        )

        # Proxy (model clients today; the single place for future consumers)
        px = raw.get("proxy", {})
        proxy = ProxyConfig(
            url=os.environ.get("CONNECTCLAW_PROXY_URL")
                or _expand_env(px.get("url", "")),
        )

        # Feishu
        fs = raw.get("feishu", {})
        feishu = FeishuConfig(
            app_id=_expand_env(
                os.environ.get("FEISHU_APP_ID", "")
                or fs.get("app_id", "")
            ),
            app_secret=_expand_env(
                os.environ.get("FEISHU_APP_SECRET", "")
                or fs.get("app_secret", "")
            ),
        )

        # Vision (supports [vision] with fallback to legacy [mimo] section)
        vision_raw = raw.get("vision") or raw.get("mimo", {})
        vision = VisionConfig(
            api_key=_expand_env(
                os.environ.get("VISION_API_KEY", "")
                or os.environ.get("MIMO_API_KEY", "")
                or vision_raw.get("api_key", "")
            ),
            base_url=os.environ.get("VISION_BASE_URL")
                or os.environ.get("MIMO_BASE_URL")
                or vision_raw.get("base_url", ""),
            model_id=os.environ.get("VISION_MODEL")
                or os.environ.get("MIMO_MODEL")
                or vision_raw.get("model_id", ""),
            # Vision is a model API: it goes through the same [proxy] as the LLM.
            proxy=os.environ.get("VISION_PROXY_URL") or proxy.url,
        )

        # Agent
        ag = raw.get("agent", {})
        agent = AgentConfig(
            cwd=os.environ.get("CONNECTCLAW_CWD")
                or ag.get("cwd", os.getcwd()),
            thinking_level=os.environ.get("CONNECTCLAW_THINKING")
                or ag.get("thinking_level", "off"),
            max_images=_safe_int(ag.get("max_images"), 5),
        )
        # Exposed-tool whitelist: only override the default when config gives a
        # non-empty list; otherwise keep AgentConfig's default (all primitives).
        ag_tools = ag.get("tools")
        if isinstance(ag_tools, list) and ag_tools:
            agent.tools = [str(x) for x in ag_tools]

        # Bash operator escape (`!` / `/bash`); empty allow-list = disabled
        ba = raw.get("bash", {})
        bash = BashConfig(
            operator_open_ids=[str(x) for x in ba.get("operator_open_ids", []) if x],
        )

        # Sandbox mode for the agent's own bash tool ("none" = full host access)
        sa = raw.get("sandbox", {})
        sandbox = SandboxConfig(
            mode=normalize_sandbox_mode(
                os.environ.get("CONNECTCLAW_SANDBOX", "")
                or str(sa.get("mode", "auto"))
            ),
        )
        if sandbox.mode != "auto":
            logger.warning(
                "[sandbox] mode=%r from config — isolation override is active",
                sandbox.mode,
            )

        # Session
        se = raw.get("session", {})
        session = SessionConfig(
            dir=os.environ.get("CONNECTCLAW_SESSIONS_DIR")
                or se.get("dir", "~/.connectclaw/sessions"),
        )

        # Skills (library scanned by the `skills` tool)
        sk = raw.get("skills", {})
        skills = SkillsConfig(
            dir=os.environ.get("CONNECTCLAW_SKILLS_DIR")
                or sk.get("dir", "~/.agents/skills"),
        )

        # RAG
        ra = raw.get("rag", {})
        rag = RAGConfig(
            # M36：env 开关与 toml 开关统一走 _safe_bool（此前 =="1" 与裸透传并存）
            enabled=_safe_bool(
                os.environ.get("CONNECTCLAW_RAG_ENABLED") or ra.get("enabled"), False
            ),
            docs_dir=os.environ.get("CONNECTCLAW_RAG_DOCS_DIR")
                or ra.get("docs_dir", ""),
            db_path=os.environ.get("CONNECTCLAW_RAG_DB_PATH")
                or ra.get("db_path", "~/.connectclaw/rag_db"),
            top_k=_safe_int(ra.get("top_k"), 20),
            top_n=_safe_int(ra.get("top_n"), 5),
        )

        # Web Search
        ws = raw.get("web_search", {})
        web_search = WebSearchConfig(
            max_chars=_safe_int(
                os.environ.get("WEB_SEARCH_MAX_CHARS")
                or ws.get("max_chars"), 8000
            ),
            timeout=_safe_int(
                os.environ.get("WEB_SEARCH_TIMEOUT")
                or ws.get("timeout"), 30
            ),
            pool_size=_safe_int(ws.get("pool_size"), 16),
            extract_model=str(ws.get("extract_model", "") or ""),
        )

        # Compaction
        co = raw.get("compaction", {})
        compaction = CompactionConfig(
            enabled=_safe_bool(co.get("enabled"), True),
            reserve_tokens=_safe_int(co.get("reserve_tokens"), 16384),
            trigger_ratio=_safe_float(co.get("trigger_ratio"), 0.8),
            keep_recent_tokens=_safe_int(co.get("keep_recent_tokens"), 20000),
            summary_max_tokens=_safe_int(co.get("summary_max_tokens"), 4096),
            memory_snapshot_max_tokens=_safe_int(co.get("memory_snapshot_max_tokens"), 2000),
        )

        # Memory
        me = raw.get("memory", {})
        memory = MemoryConfig(
            enabled=_safe_bool(
                os.environ.get("CONNECTCLAW_MEMORY_ENABLED") or me.get("enabled"), True
            ),
            db_path=os.environ.get("CONNECTCLAW_MEMORY_DB_PATH")
            or me.get("db_path", "~/.connectclaw/memory.db"),
            extract_after_turn=_safe_bool(me.get("extract_after_turn"), True),
            extract_min_turns=_safe_int(me.get("extract_min_turns"), 3),
            extract_interval_turns=_safe_int(me.get("extract_interval_turns"), 5),
            max_context_tokens=_safe_int(me.get("max_context_tokens"), 2000),
            recency_threshold_days=_safe_int(me.get("recency_threshold_days"), 7),
            use_embeddings=_safe_bool(me.get("use_embeddings"), True),
            dream_interval_hours=_safe_float(me.get("dream_interval_hours"), 24.0),
            decay_halflife_days=_safe_float(me.get("decay_halflife_days"), 30.0),
            consolidation_enabled=_safe_bool(me.get("consolidation_enabled"), True),
        )

        return cls(
            llm=llm,
            feishu=feishu,
            vision=vision,
            proxy=proxy,
            agent=agent,
            bash=bash,
            sandbox=sandbox,
            session=session,
            skills=skills,
            rag=rag,
            web_search=web_search,
            compaction=compaction,
            memory=memory,
        )
