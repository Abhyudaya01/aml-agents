"""Window scoring and the alert record.

    combined_risk_score = RULE_WEIGHT * min(1, sum of fired rule weights)
                        + ANOMALY_WEIGHT * anomaly percentile (vs. train)

Weights are set by hand and documented, not tuned. An alert is raised when
combined_risk_score >= thresholds.alert_threshold.
"""

from datetime import date

import duckdb
import numpy as np
from pydantic import BaseModel

from aml.alerts.rules import rule_score_sql, triggered_rules_sql
from aml.alerts.thresholds import RuleThresholds
from aml.alerts.windows import WINDOW_DAYS

RULE_WEIGHT = 0.6
ANOMALY_WEIGHT = 0.4


class Alert(BaseModel):
    alert_id: str
    bank: str
    account: str
    window_start: date
    window_end: date  # last day of the window, inclusive
    triggered_rules: list[str]
    rule_txn_ids: list[str]
    anomaly_score: float
    anomaly_pct: float
    combined_risk_score: float


def budget_threshold(scores: np.ndarray, budget_per_1k: float) -> float:
    """Smallest cutoff such that at most budget_per_1k / 1000 of `scores` are >= it.

    Ties at the boundary are excluded rather than letting the budget overflow.
    """
    k = int(len(scores) * budget_per_1k / 1000)
    if k <= 0:
        return float("inf")
    desc = np.sort(scores)[::-1]
    tau = desc[k - 1]
    if k < len(desc) and desc[k] == tau:  # tie straddles the budget: move above it
        above = desc[desc > tau]
        tau = above.min() if above.size else float("inf")
    return float(tau)


def combined_score_sql(th: RuleThresholds) -> str:
    """Combined risk score over features `f` joined with anomaly_scores `a`."""
    return f"{RULE_WEIGHT} * {rule_score_sql(th)} + {ANOMALY_WEIGHT} * a.anomaly_pct"


def score_windows(conn: duckdb.DuckDBPyConnection, th: RuleThresholds) -> None:
    """Create window_scores for every account-window (train and test)."""
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE window_scores AS
        SELECT f.bank, f.account, f.window_start, f.split,
               {triggered_rules_sql(th)} AS triggered_rules,
               {rule_score_sql(th)} AS rule_score,
               a.anomaly_score, a.anomaly_pct,
               {combined_score_sql(th)} AS combined_risk_score
        FROM account_window_features f
        JOIN anomaly_scores a USING (bank, account, window_start)
        """
    )


def alerts_select_sql(score_col: str, tau: float) -> str:
    """Alerted windows for a score column of window_scores, with their rule evidence."""
    return f"""
        SELECT s.*, coalesce(h.rule_txn_ids, CAST([] AS VARCHAR[])) AS rule_txn_ids
        FROM window_scores s
        LEFT JOIN (
            SELECT bank, account, window_start,
                   list_sort(list_distinct(flatten(list(txn_ids)))) AS rule_txn_ids
            FROM rule_hits GROUP BY ALL
        ) h USING (bank, account, window_start)
        WHERE s.{score_col} >= {tau}
    """


def build_alerts(conn: duckdb.DuckDBPyConnection, th: RuleThresholds) -> int:
    """Create the alerts table. Requires rule_hits and window_scores."""
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE alerts AS
        SELECT 'ALR-' || substr(
                   sha256(bank || ':' || account || ':' || CAST(window_start AS VARCHAR)), 1, 12
               ) AS alert_id,
               bank, account, window_start,
               CAST(window_start + {WINDOW_DAYS - 1} AS DATE) AS window_end,
               triggered_rules, rule_txn_ids, anomaly_score, anomaly_pct, combined_risk_score,
               split
        FROM ({alerts_select_sql("combined_risk_score", th.alert_threshold)})
        ORDER BY combined_risk_score DESC
        """
    )
    return conn.execute("SELECT count(*) FROM alerts").fetchone()[0]


def fetch_alert(conn: duckdb.DuckDBPyConnection, alert_id: str) -> Alert:
    cols = list(Alert.model_fields)
    row = conn.execute(
        f"SELECT {', '.join(cols)} FROM alerts WHERE alert_id = ?", [alert_id]
    ).fetchone()
    if row is None:
        raise KeyError(alert_id)
    return Alert(**dict(zip(cols, row, strict=True)))
