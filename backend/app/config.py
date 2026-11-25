from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class KeyEntry:
    key_id: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Settings:
    key_pool: List[KeyEntry]
    default_model: str
    database_url: str
    allowed_origins: List[str]
    enable_failure_simulation: bool
    default_system_prompt: str
    cache_type: str
    cache_ttl_seconds: int
    cache_dir: str


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    raw_keys = os.getenv("LLM_API_KEYS", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    default_model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash").strip() or "gemini/gemini-2.0-flash"
    raw_models = os.getenv("LLM_MODELS", "").strip()
    models = [m.strip() for m in raw_models.split(",") if m.strip()] if raw_models else []

    key_pool: List[KeyEntry] = []
    for idx, api_key in enumerate(keys, start=1):
        if models:
            model = models[idx - 1] if idx - 1 < len(models) else models[-1]
        else:
            model = default_model
        key_pool.append(KeyEntry(key_id=f"key_{idx}", api_key=api_key, model=model))

    database_url = os.getenv("DATABASE_URL", "sqlite:///./app.db").strip() or "sqlite:///./app.db"

    raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    if not allowed_origins:
        allowed_origins = ["http://localhost:3000"]

    enable_failure_simulation = _parse_bool(os.getenv("ENABLE_FAILURE_SIMULATION"), default=False)
    default_system_prompt = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a helpful, concise AI assistant.",
    ).strip()

    cache_type = os.getenv("CACHE_TYPE", "local").strip().lower() or "local"
    if cache_type not in {"local", "disk"}:
        cache_type = "local"

    try:
        cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    except ValueError:
        cache_ttl_seconds = 3600

    cache_dir = os.getenv("CACHE_DIR", "/tmp/litellm_cache").strip() or "/tmp/litellm_cache"

    return Settings(
        key_pool=key_pool,
        default_model=default_model,
        database_url=database_url,
        allowed_origins=allowed_origins,
        enable_failure_simulation=enable_failure_simulation,
        default_system_prompt=default_system_prompt,
        cache_type=cache_type,
        cache_ttl_seconds=cache_ttl_seconds,
        cache_dir=cache_dir,
    )
