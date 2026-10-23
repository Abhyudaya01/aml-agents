"""Application settings, loaded from environment variables and `.env` (prefix `AML_`)."""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["fake", "anthropic", "bedrock", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AML_", env_file=".env", extra="ignore")

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    duckdb_path: Path = Path("data/aml.duckdb")
    audit_db_path: Path = Path("data/audit.sqlite")

    llm_provider: LLMProvider = "fake"
    model_id: str | None = None
    aws_region: str | None = None

    # Placeholder until the HI-Small date range is inspected in the data phase.
    train_test_split_date: date = date(2022, 9, 10)


@lru_cache
def get_settings() -> Settings:
    return Settings()
