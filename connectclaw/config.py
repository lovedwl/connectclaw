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
        return os.environ.get(var_name, "")
    return pattern.sub(replace, value)


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
        logger.warning("Failed to load %s: %s", path, e)
        return {}


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


@dataclass
class FeishuConfig:
    app_id: str = ""
    app_secret: str = ""


@dataclass
class VisionConfig:
    api_key: str = ""
    base_url: str = ""
    model_id: str = ""


@dataclass
class AgentConfig:
    cwd: str = field(default_factory=os.getcwd)
    thinking_level: str = "off"
    max_images: int = 5
    # Which base primitives are exposed DIRECTLY to the main agent (a whitelist
    # from the tool registry). Named agents + dynamic tools are NOT here — they
    # are reached through the single `agents` meta-tool. Default = all base
    # primitives (current behaviour). Unknown names are dropped with a warning;
    # an empty resolved set falls back to ['read','bash'].
    tools: list[str] = field(
        default_factory=lambda: [
            "read", "write", "hash_read", "hash_edit",
            "bash", "web_search", "web_fetch", "image_analyze",
        ]
    )
    # Idle seconds before a stateful scripted-tool session process is reaped.
    tool_session_idle_timeout: int = 300


@dataclass
class SessionConfig:
    dir: str = "~/.connectclaw/sessions"


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


@dataclass
class CompactionConfig:
    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000


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
    agent: AgentConfig = field(default_factory=AgentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
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
        )

        # Agent
        ag = raw.get("agent", {})
        agent = AgentConfig(
            cwd=os.environ.get("CONNECTCLAW_CWD")
                or ag.get("cwd", os.getcwd()),
            thinking_level=os.environ.get("CONNECTCLAW_THINKING")
                or ag.get("thinking_level", "off"),
            max_images=int(ag.get("max_images", 5)),
        )
        # Exposed-tool whitelist: only override the default when config gives a
        # non-empty list; otherwise keep AgentConfig's default (all primitives).
        ag_tools = ag.get("tools")
        if isinstance(ag_tools, list) and ag_tools:
            agent.tools = [str(x) for x in ag_tools]
        agent.tool_session_idle_timeout = int(
            ag.get("tool_session_idle_timeout", agent.tool_session_idle_timeout)
        )

        # Session
        se = raw.get("session", {})
        session = SessionConfig(
            dir=os.environ.get("CONNECTCLAW_SESSIONS_DIR")
                or se.get("dir", "~/.connectclaw/sessions"),
        )

        # RAG
        ra = raw.get("rag", {})
        rag = RAGConfig(
            enabled=os.environ.get("CONNECTCLAW_RAG_ENABLED", "") == "1"
                or ra.get("enabled", False),
            docs_dir=os.environ.get("CONNECTCLAW_RAG_DOCS_DIR")
                or ra.get("docs_dir", ""),
            db_path=os.environ.get("CONNECTCLAW_RAG_DB_PATH")
                or ra.get("db_path", "~/.connectclaw/rag_db"),
            top_k=int(ra.get("top_k", 20)),
            top_n=int(ra.get("top_n", 5)),
        )

        # Web Search
        ws = raw.get("web_search", {})
        web_search = WebSearchConfig(
            max_chars=int(
                os.environ.get("WEB_SEARCH_MAX_CHARS", "")
                or ws.get("max_chars", 8000)
            ),
            timeout=int(
                os.environ.get("WEB_SEARCH_TIMEOUT", "")
                or ws.get("timeout", 30)
            ),
            pool_size=int(ws.get("pool_size", 16)),
        )

        # Compaction
        co = raw.get("compaction", {})
        compaction = CompactionConfig(
            enabled=co.get("enabled", True),
            reserve_tokens=int(co.get("reserve_tokens", 16384)),
            keep_recent_tokens=int(co.get("keep_recent_tokens", 20000)),
        )

        # Memory
        me = raw.get("memory", {})
        memory = MemoryConfig(
            enabled=os.environ.get("CONNECTCLAW_MEMORY_ENABLED", "") != "0"
            and me.get("enabled", True),
            db_path=os.environ.get("CONNECTCLAW_MEMORY_DB_PATH")
            or me.get("db_path", "~/.connectclaw/memory.db"),
            extract_after_turn=me.get("extract_after_turn", True),
            extract_min_turns=int(me.get("extract_min_turns", 3)),
            extract_interval_turns=int(me.get("extract_interval_turns", 5)),
            max_context_tokens=int(me.get("max_context_tokens", 2000)),
            recency_threshold_days=int(me.get("recency_threshold_days", 7)),
            use_embeddings=me.get("use_embeddings", True),
            dream_interval_hours=float(me.get("dream_interval_hours", 24.0)),
            decay_halflife_days=float(me.get("decay_halflife_days", 30.0)),
            consolidation_enabled=me.get("consolidation_enabled", True),
        )

        return cls(
            llm=llm,
            feishu=feishu,
            vision=vision,
            agent=agent,
            session=session,
            rag=rag,
            web_search=web_search,
            compaction=compaction,
            memory=memory,
        )
