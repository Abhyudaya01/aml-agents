"""Detection rules over account-window features.

Each rule is a SQL condition on `account_window_features` (whether it fires) plus
an evidence query (which txn_ids triggered it). The same conditions drive
scoring, tuning, and the stored `rule_hits`, so there is one definition per rule.
"""

from collections.abc import Callable
from dataclasses import dataclass

import duckdb

from aml.alerts.features import STRUCTURING_BAND, STRUCTURING_THRESHOLD_USD
from aml.alerts.thresholds import RuleThresholds


@dataclass(frozen=True)
class Rule:
    rule_id: str
    weight: float  # hand-set severity, documented in PROGRESS.md; not tuned
    condition: Callable[[RuleThresholds], str | None]  # SQL over features (alias f)
    evidence: str  # SQL selecting (bank, account, window_start, txn_id) for fired windows


def _legs_evidence(where: str) -> str:
    return f"""
        SELECT l.bank, l.account, l.window_start, l.txn_id
        FROM legs l JOIN fired f USING (bank, account, window_start)
        WHERE {where}
    """


def _at_least(column: str, value: float | None) -> str | None:
    return None if value is None else f"f.{column} >= {value}"


def _velocity(th: RuleThresholds) -> str | None:
    if th.velocity_min_24h is None or th.velocity_min_burst is None:
        return None
    return f"f.max_24h >= {th.velocity_min_24h} AND f.burst_ratio >= {th.velocity_min_burst}"


_LOW = STRUCTURING_THRESHOLD_USD * (1 - STRUCTURING_BAND)

RULES: tuple[Rule, ...] = (
    Rule(
        "FAN_IN",
        0.5,
        lambda th: _at_least("cp_in", th.fan_in_min_cp),
        _legs_evidence("l.direction = 'in'"),
    ),
    Rule(
        "FAN_OUT",
        0.5,
        lambda th: _at_least("cp_out", th.fan_out_min_cp),
        _legs_evidence("l.direction = 'out'"),
    ),
    Rule(
        "STRUCTURING",
        0.6,
        lambda th: _at_least("n_near_threshold", th.structuring_min_count),
        _legs_evidence(f"l.amount_usd >= {_LOW} AND l.amount_usd < {STRUCTURING_THRESHOLD_USD}"),
    ),
    Rule(
        "VELOCITY_SPIKE",
        0.4,
        _velocity,
        # Same inclusive range as the rolling count in features.py.
        _legs_evidence("l.ts BETWEEN f.peak_24h_end - INTERVAL 24 HOURS AND f.peak_24h_end"),
    ),
    Rule(
        "PASS_THROUGH",
        0.7,
        lambda th: _at_least("n_pass_through", th.pass_through_min_count),
        """
        SELECT p.bank, p.account, p.window_start, unnest(p.pass_through_txn_ids) AS txn_id
        FROM pass_through p JOIN fired f USING (bank, account, window_start)
        """,
    ),
)


def rule_score_sql(th: RuleThresholds) -> str:
    """min(1, sum of weights of fired rules), over features aliased `f`."""
    terms = [
        f"CASE WHEN {cond} THEN {r.weight} ELSE 0 END"
        for r in RULES
        if (cond := r.condition(th)) is not None
    ]
    return f"least(1.0, {' + '.join(terms) or '0'})"


def triggered_rules_sql(th: RuleThresholds) -> str:
    """List of fired rule_ids, over features aliased `f`."""
    items = [
        f"CASE WHEN {cond} THEN '{r.rule_id}' END"
        for r in RULES
        if (cond := r.condition(th)) is not None
    ]
    if not items:
        return "CAST([] AS VARCHAR[])"
    return f"list_filter([{', '.join(items)}], lambda x: x IS NOT NULL)"


def build_rule_hits(conn: duckdb.DuckDBPyConnection, th: RuleThresholds) -> dict[str, int]:
    """Create rule_hits(bank, account, window_start, rule_id, txn_ids). Returns hits per rule."""
    conn.execute(
        "CREATE OR REPLACE TABLE rule_hits (bank VARCHAR, account VARCHAR, window_start DATE, "
        "rule_id VARCHAR, txn_ids VARCHAR[])"
    )
    counts = {}
    for rule in RULES:
        cond = rule.condition(th)
        if cond is None:
            counts[rule.rule_id] = 0
            continue
        conn.execute(
            f"CREATE OR REPLACE TEMP TABLE fired AS "
            f"SELECT * FROM account_window_features f WHERE {cond}"
        )
        conn.execute(
            f"""
            INSERT INTO rule_hits
            SELECT bank, account, window_start, '{rule.rule_id}',
                   list(DISTINCT txn_id ORDER BY txn_id)
            FROM ({rule.evidence}) GROUP BY bank, account, window_start
            """
        )
        counts[rule.rule_id] = conn.execute("SELECT count(*) FROM fired").fetchone()[0]
    conn.execute("DROP TABLE IF EXISTS fired")
    return counts
