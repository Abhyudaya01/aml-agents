"""Application settings, loaded from environment variables and `.env` (prefix `AML_`)."""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["fake", "anthropic", "bedrock", "openai"]

# Anthropic first-party API list prices (USD per 1M tokens), as of 2026-10.
# Bedrock and OpenAI prices differ and must be supplied via AML_LLM_PRICE_OVERRIDES.
# Haiku 5.5's price applies to prompts up to 100K tokens.
DEFAULT_LLM_PRICES: dict[str, dict[str, float]] = {
    "fake": {"input_per_mtok": 0.0, "output_per_mtok": 0.0},
    "claude-opus-5-5": {"input_per_mtok": 4.0, "output_per_mtok": 20.0},
    "claude-sonnet-5-5": {"input_per_mtok": 2.0, "output_per_mtok": 10.0},
    "claude-haiku-5-5": {"input_per_mtok": 0.10, "output_per_mtok": 0.50},
    "claude-fable-5-1": {"input_per_mtok": 10.0, "output_per_mtok": 50.0},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AML_", env_file=".env", extra="ignore")

    raw_data_dir: Path = Path("data/raw")
    transactions_file: str = "HI-Small_Trans.csv"
    patterns_file: str = "HI-Small_Patterns.txt"
    processed_data_dir: Path = Path("data/processed")
    duckdb_path: Path = Path("data/aml.duckdb")
    audit_db_path: Path = Path("data/audit.sqlite")

    llm_provider: LLMProvider = "fake"
    model_id: str | None = None  # None -> provider default (anthropic: claude-opus-5-5)
    aws_region: str | None = None
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 16_000
    llm_max_retries: int = 2
    # USD per 1M tokens. Anthropic first-party list prices; add others via overrides.
    llm_prices: dict[str, dict[str, float]] = DEFAULT_LLM_PRICES
    llm_price_overrides: dict[str, dict[str, float]] = {}
    pii_vault_path: Path = Path("data/pii_vault.sqlite")

    # Train = before this date (midnight). Sep 1-7 is 73.5% of HI-Small rows; see PROGRESS.md.
    train_test_split_date: date = date(2022, 9, 8)
    customer_seed: int = 42

    artifacts_dir: Path = Path("artifacts")
    alert_budget_per_1k_windows: float = 10.0

    @property
    def thresholds_path(self) -> Path:
        return self.artifacts_dir / "thresholds.json"

    @property
    def anomaly_model_path(self) -> Path:
        return self.artifacts_dir / "isolation_forest.joblib"

    @property
    def transactions_path(self) -> Path:
        return self.raw_data_dir / self.transactions_file

    @property
    def patterns_path(self) -> Path:
        return self.raw_data_dir / self.patterns_file


@lru_cache
def get_settings() -> Settings:
    return Settings()
