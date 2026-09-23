"""Model profile registry — the escape-hatch source of truth for /model.

A model profile = {name, base_url, model_id, api_key, reasoning,
context_window, max_tokens, desc}. Profiles live in ``~/.connectclaw/models.toml``
(own file on purpose: config.toml is hand-edited with many comments, a full
rewrite would destroy them). The ACTIVE model is still the ``[llm]`` section in
config.toml; ``set_active`` updates just that block surgically so a restart
keeps the same selection.

``api_key`` may be ``$ENV_VAR``-style (expanded via config._expand_env at use).
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import tomli_w

from connectclaw.config import _expand_env

DEFAULT_REGISTRY = os.path.expanduser("~/.connectclaw/models.toml")

# [llm] section regexp: a line exactly "[llm]" followed by non-section lines.
_LLM_BLOCK_RE = re.compile(r"^\[llm\]\n(?:[^\[\n][^\n]*\n)*", re.M)

# Markdown link pasted/typed into chat: [label](href). Feishu turns a pasted URL
# into a rich-text link element, which the channel flattens back to markdown, so
# this is common input for base_url / verify, not an exotic edge case.
_MD_LINK_RE = re.compile(r"\[(?P<label>[^\]]*)\]\((?P<href>[^)]*)\)")
# A bare host[:port][/path] — safe to prefix with a scheme. Deliberately
# rejects free-form words ("not-a-url") so callers keep their validation.
_BARE_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+(:\d+)?(/[^\s]*)?$")


def normalize_url(raw: str) -> str:
    """Normalize an endpoint string from chat into a usable URL.

    Handles the two shapes users actually produce: a pasted markdown link
    (``[https://gw](https://gw)`` — no HTTP client accepts that string) and a
    bare host (``gw.example.com`` → ``https://gw.example.com``). The trailing
    slash is dropped so ``f"{base}/v1"`` never doubles up. Returns "" for
    empty input; a non-URL string is returned as-is (minus trimming) so
    callers can still reject it.
    """
    s = (raw or "").strip().strip("`").strip()
    link = _MD_LINK_RE.search(s)
    if link:
        s = (link.group("href") or link.group("label")).strip()
    s = s.strip().strip("<>").strip().rstrip("/")
    if s and "://" not in s and _BARE_HOST_RE.match(s) and "." in s.split("/")[0]:
        s = f"https://{s}"
    return s


def mask_key(api_key: str) -> str:
    """Short fingerprint of a key for display — never the whole secret."""
    k = (api_key or "").strip()
    if not k:
        return "(无 key)"
    if len(k) <= 12:
        return k[:3] + "…"
    return f"{k[:6]}…{k[-4:]}"


@dataclass
class ModelProfile:
    name: str
    base_url: str
    model_id: str
    api_key: str = ""
    reasoning: bool = True
    context_window: int = 65536
    max_tokens: int = 8192
    desc: str = ""
    provider: str = ""
    """Display grouping for the /model picker — in practice the KEY identity
    (same base_url with different keys = different providers, e.g. USTC's
    blocked vs working key). Empty → derived from the base_url host."""

    def resolved_api_key(self) -> str:
        return _expand_env(self.api_key or "")

    def display_provider(self) -> str:
        if self.provider:
            return self.provider
        host = urlparse(self.base_url).hostname if self.base_url else ""
        return host or "未分组"


class ModelsStore:
    """CRUD over ~/.connectclaw/models.toml + activation into config.toml."""

    def __init__(self, registry_path: str | None = None, config_path: str | None = None):
        from connectclaw.config import _find_config_path

        self._registry = Path(registry_path or DEFAULT_REGISTRY)
        self._config_path = config_path or _find_config_path()
        self._profiles: dict[str, ModelProfile] | None = None

    # ── Registry file ───────────────────────────────────────

    @property
    def _path(self) -> str:
        return str(self._registry)

    def _load(self) -> dict[str, ModelProfile]:
        try:
            with open(self._path, "rb") as f:
                raw = tomllib.load(f)
        except (FileNotFoundError, tomllib.TOMLDecodeError):
            raw = {}
        profiles: dict[str, ModelProfile] = {}
        for item in raw.get("profiles") or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            try:
                profiles[item["name"]] = ModelProfile(
                    name=str(item["name"]),
                    base_url=str(item.get("base_url", "")),
                    model_id=str(item.get("model_id", "")),
                    api_key=str(item.get("api_key", "")),
                    reasoning=bool(item.get("reasoning", True)),
                    context_window=int(item.get("context_window", 65536)),
                    max_tokens=int(item.get("max_tokens", 8192)),
                    desc=str(item.get("desc", "")),
                    provider=str(item.get("provider", "")),
                )
            except (TypeError, ValueError):
                continue
        return profiles

    def _reload(self) -> None:
        self._profiles = self._load()

    def _save(self) -> None:
        entries = {p.name: asdict(p) for p in self._profiles.values()}
        text = tomli_w.dumps({"profiles": [entries[n] for n in sorted(entries)]})
        self._registry.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._registry.write_text(text, encoding="utf-8")
            os.chmod(self._path, 0o600)
        except OSError:
            pass

    # ── CRUD ───────────────────────────────────────────────

    def list(self) -> list[ModelProfile]:
        if self._profiles is None:
            self._reload()
        return list(self._profiles.values())

    def get(self, name: str) -> ModelProfile | None:
        if self._profiles is None:
            self._reload()
        return self._profiles.get(name)

    def save(self, profile: ModelProfile) -> str:
        """Save/upsert a profile, dedup-ing the auto name if needed."""
        if self._profiles is None:
            self._reload()
        name = profile.name.strip()
        if not name:
            # Auto-name from model_id, dedup with -2/-3...
            base = (profile.model_id or "model").strip() or "model"
            name, i = base, 2
            while name in self._profiles:
                name = f"{base}-{i}"
                i += 1
            profile.name = name
        self._profiles[profile.name] = profile
        self._save()
        return profile.name

    def delete(self, name: str) -> bool:
        if self._profiles is None:
            self._reload()
        if self._profiles.pop(name, None) is None:
            return False
        self._save()
        return True

    # ── Activation (config.toml [llm] surgical replace) ────

    def set_active(self, profile: ModelProfile) -> None:
        """Persist the profile into the [llm] section of config.toml without
        touching any other section or comment."""
        self.save(profile)  # ensure it's in the registry too
        block = "[llm]\n"
        for k in ("api_key", "base_url", "model_id",
                  "context_window", "max_tokens", "reasoning"):
            v = getattr(profile, k)
            if isinstance(v, bool):
                block += f"{k} = {'true' if v else 'false'}\n"
            elif isinstance(v, int):
                block += f"{k} = {v}\n"
            else:
                if k == "api_key" and not v:
                    continue
                block += f'{k} = "{str(v).replace(chr(34), chr(39))}"\n'
        self._write_llm_block(block)

    def _write_llm_block(self, block: str) -> None:
        if not self._config_path or not os.path.isfile(self._config_path):
            return
        try:
            with open(self._config_path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return
        if _LLM_BLOCK_RE.search(text):
            text = _LLM_BLOCK_RE.sub(block, text, count=1)
        else:
            text = text.rstrip() + "\n\n" + block
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass
