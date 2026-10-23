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
    transactions_file: str = "HI-Small_Trans.csv"
    patterns_file: str = "HI-Small_Patterns.txt"
    processed_data_dir: Path = Path("data/processed")
    duckdb_path: Path = Path("data/aml.duckdb")
    audit_db_path: Path = Path("data/audit.sqlite")

    llm_provider: LLMProvider = "fake"
    model_id: str | None = None
    aws_region: str | None = None

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
