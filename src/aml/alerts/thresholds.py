"""Rule thresholds: the only output of tuning that detection code is allowed to see.

Tuning (in the evaluation package) writes `artifacts/thresholds.json`; alert
generation reads it. A threshold of `None` disables that rule.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel


class RuleThresholds(BaseModel):
    fan_in_min_cp: float | None
    fan_out_min_cp: float | None
    structuring_min_count: float | None
    velocity_min_24h: float | None
    velocity_min_burst: float | None
    pass_through_min_count: float | None
    # Combined-score cutoff, chosen so the train split meets the alert budget.
    alert_threshold: float
    budget_per_1k_windows: float
    tuned_on: str = "train"


def save_thresholds(th: RuleThresholds, path: Path) -> str:
    """Write thresholds as JSON; returns the file's sha256 for provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(th.model_dump_json(indent=2) + "\n")
    return thresholds_sha256(path)


def load_thresholds(path: Path) -> RuleThresholds:
    return RuleThresholds.model_validate_json(path.read_text())


def thresholds_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
