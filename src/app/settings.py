"""Typed process settings. Fail closed if a required variable is missing.

`.env.example` lists every field in this module, with placeholders only.
Call `Settings(_env_file=None)` in tests to ignore a local `.env` file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "staging", "production"]


class Settings(BaseSettings):
    """Runtime settings loaded from the environment (and optional `.env`)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Environment
    log_level: str = "INFO"
    llm_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None
    aws_region: str | None = None
    aws_account_id: str | None = Field(default=None, description="Use <ACCOUNT_ID> in examples.")
    database_url: SecretStr | None = None

    @field_validator(
        "llm_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        "database_url",
        "langfuse_host",
        "aws_region",
        "aws_account_id",
        mode="before",
    )
    @classmethod
    def empty_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def require_llm_key_outside_local(self) -> Self:
        if self.app_env != "local" and self.llm_api_key is None:
            raise ValueError("LLM_API_KEY is required when APP_ENV is staging or production")
        return self

    def secret_values(self) -> list[str]:
        """Plain secret strings, for redaction only. Never log these."""
        values: list[str] = []
        secrets = (
            self.llm_api_key,
            self.langfuse_public_key,
            self.langfuse_secret_key,
            self.database_url,
        )
        for item in secrets:
            if item is not None:
                plain = item.get_secret_value()
                if plain:
                    values.append(plain)
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
