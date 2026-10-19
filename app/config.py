"""Runtime configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    manifest_path: Path
    db_path: Path
    app_today: date | None
    llm_provider: str
    anthropic_model: str
    llm_effort: str
    llm_fallbacks: str
    llm_timeout_s: float
    request_deadline_s: float
    embedding_model: str
    model_cache_dir: str
    log_level: str
    log_file: str | None
    extra: dict = field(default_factory=dict)

    def today(self) -> date:
        """APP_TODAY if set, otherwise the system date."""
        return self.app_today or date.today()


def load_settings() -> Settings:
    app_today = _env("APP_TODAY")
    return Settings(
        corpus_dir=Path(_env("CORPUS_DIR", "./data/corpus")),
        manifest_path=Path(_env("MANIFEST_PATH", "./data/manifest.json")),
        db_path=Path(_env("DB_PATH", "./data/veltrix.db")),
        app_today=date.fromisoformat(app_today) if app_today else None,
        llm_provider=_env("LLM_PROVIDER", "anthropic").lower(),
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-opus-5-5"),
        llm_effort=_env("LLM_EFFORT", "low"),
        llm_fallbacks=_env("LLM_FALLBACKS", "default"),
        llm_timeout_s=float(_env("LLM_TIMEOUT_S", "20")),
        request_deadline_s=float(_env("REQUEST_DEADLINE_S", "27")),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        model_cache_dir=_env("MODEL_CACHE_DIR", "./.cache/models"),
        log_level=_env("LOG_LEVEL", "INFO"),
        log_file=_env("LOG_FILE"),
    )
