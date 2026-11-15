"""Settings from the environment (pydantic-settings). Nothing secret lives in git."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JWT_SECRET = "dev-only-insecure-secret-change-me"  # noqa: S105 (documented sentinel)
DEFAULT_DEMO_PASSWORD = "demo-password-123"  # noqa: S105 (non-production only)
MIN_PRODUCTION_SECRET_LENGTH = 32


class ConfigError(RuntimeError):
    """The configuration is unsafe; the app refuses to start."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    env: Literal["development", "test", "production"] = "development"
    jwt_secret: str = DEFAULT_JWT_SECRET
    demo_password: str = ""
    session_minutes: int = 60
    cors_origins: str = "http://localhost:3000"
    model_dir: Path = Path("ml/models")
    processed_data_path: Path = Path("data/processed/brfss_clean.parquet")

    @model_validator(mode="after")
    def _refuse_unsafe_production(self) -> Settings:
        if self.env != "production":
            return self
        secret = self.jwt_secret
        if not secret or secret == DEFAULT_JWT_SECRET:
            raise ConfigError("ENV=production requires JWT_SECRET to be set to a real secret")
        if len(secret) < MIN_PRODUCTION_SECRET_LENGTH:
            raise ConfigError(f"JWT_SECRET must be at least {MIN_PRODUCTION_SECRET_LENGTH} chars")
        if not self.demo_password:
            raise ConfigError("ENV=production requires DEMO_PASSWORD to be set explicitly")
        return self

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def effective_demo_password(self) -> str:
        return self.demo_password or DEFAULT_DEMO_PASSWORD

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def models_path(self) -> Path:
        return _resolve(self.model_dir)

    @property
    def processed_path(self) -> Path:
        return _resolve(self.processed_data_path)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path
