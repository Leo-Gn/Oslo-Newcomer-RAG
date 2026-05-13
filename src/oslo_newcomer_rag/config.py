from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


AppEnv = Literal["development", "test", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Oslo Newcomer Assistant"
    app_env: AppEnv = "development"
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)

    database_url: str | None = None

    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = Field(default=None, gt=0)

    request_body_limit_bytes: int = Field(default=131_072, ge=4096, le=1_048_576)
    rate_limit_enabled: bool = True
    chat_rate_limit_per_minute: int = Field(default=20, ge=1, le=600)
    feedback_rate_limit_per_minute: int = Field(default=60, ge=1, le=1200)

    @model_validator(mode="after")
    def require_production_settings(self) -> "Settings":
        if self.llm_base_url:
            _validate_llm_base_url(self.llm_base_url, production=self.app_env == "production")
        if self.database_url and self.app_env == "production":
            _validate_production_database_url(self.database_url)

        if self.app_env != "production":
            return self

        required = {
            "DATABASE_URL": self.database_url,
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_API_KEY": self.llm_api_key,
            "LLM_MODEL": self.llm_model,
            "EMBEDDING_MODEL": self.embedding_model,
            "EMBEDDING_DIM": self.embedding_dim,
        }
        missing = [name for name, value in required.items() if value in (None, "")]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing required production settings: {joined}")

        return self

    @property
    def has_database_config(self) -> bool:
        return bool(self.database_url)

    @property
    def has_llm_config(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)


def _validate_llm_base_url(value: str, *, production: bool) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LLM_BASE_URL must be an http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("LLM_BASE_URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("LLM_BASE_URL must not contain a query string or fragment")
    if production and parsed.scheme != "https":
        raise ValueError("LLM_BASE_URL must use https in production")


def _validate_production_database_url(value: str) -> None:
    parsed = urlparse(value)
    if not parsed.scheme.startswith("postgresql") or not parsed.netloc:
        raise ValueError("DATABASE_URL must be a PostgreSQL URL in production")


@lru_cache
def get_settings() -> Settings:
    return Settings(_env_file=".env")
