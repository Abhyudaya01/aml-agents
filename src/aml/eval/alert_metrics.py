"""Alert-level evaluation against ground truth.

A positive unit is an (account, window) containing at least one laundering
transaction in which that account is sender or receiver. An alert is a true
positive if its unit is positive.

Metrics are reported per slice:
- train:   in-sample (thresholds were tuned here); for reference only
- test:    all test windows (Sep 8-14 and the sparse Sep 15+ tail)
- test_w1: the first test window only, which excludes the tail
"""

from datetime import date

import duckdb
from pydantic import BaseModel

from aml.alerts.records import alerts_select_sql, budget_threshold
from aml.alerts.windows import window_start_sql
from aml.eval.labels import EVAL_SCHEMA

POSITIVE_UNITS = f"{EVAL_SCHEMA}.positive_units"


class AlertMetrics(BaseModel):
    variant: str
    slice: str
    alerts: int
    true_positives: int
    precision: float
    positive_units: int
    recall: float
    active_accounts: int
    alerts_per_1k_accounts: float
    account_windows: int
    alerts_per_1k_windows: float
    laundering_txns: int
    txn_recall_cited: float  # laundering txns listed in some alert's rule_txn_ids
    txn_recall_window: float  # laundering txns touching an alerted account-window


class TypologyRecall(BaseModel):
    typology: str
    units: int
    units_alerted: int
    unit_recall: float
    attempts: int
    attempts_detected: int
    attempt_recall: float


class RuleBreakdown(BaseModel):
    rule_id: str
    alerts: int
    true_positives: int
    precision: float


def slice_conditions(split_date: date) -> dict[str, str]:
    d = f"DATE '{split_date.isoformat()}'"
    return {
        "train": f"window_start < {d}",
        "test": f"window_start >= {d}",
        "test_w1": f"window_start = {d}",
    }


def build_positive_units(conn: duckdb.DuckDBPyConnection, split_date: date) -> int:
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {POSITIVE_UNITS} AS
        WITH pos AS (
            SELECT t.*, {window_start_sql("t.timestamp", split_date)} AS window_start
            FROM transactions t JOIN {EVAL_SCHEMA}.txn_labels l USING (txn_id)
            WHERE l.is_laundering
        )
        SELECT from_bank AS bank, from_account AS account, window_start FROM pos
        UNION
        SELECT to_bank, to_account, window_start FROM pos
        """
    )
    return conn.execute(f"SELECT count(*) FROM {POSITIVE_UNITS}").fetchone()[0]


def _laundering_txns_sql(split_date: date) -> str:
    return f"""
        SELECT t.txn_id, t.from_bank, t.from_account, t.to_bank, t.to_account,
               {window_start_sql("t.timestamp", split_date)} AS window_start
        FROM transactions t JOIN {EVAL_SCHEMA}.txn_labels l USING (txn_id)
        WHERE l.is_laundering
    """


def evaluate_slice(
    conn: duckdb.DuckDBPyConnection,
    alerts_table: str,
    variant: str,
    slice_name: str,
    cond: str,
    split_date: date,
) -> AlertMetrics:
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    n_alerts = q(f"SELECT count(*) FROM {alerts_table} WHERE {cond}")
    tp = q(
        f"SELECT count(*) FROM {alerts_table} a JOIN {POSITIVE_UNITS} p "
        f"USING (bank, account, window_start) WHERE a.{cond}"
    )
    positives = q(f"SELECT count(*) FROM {POSITIVE_UNITS} WHERE {cond}")
    n_windows, n_accounts = conn.execute(
        f"SELECT count(*), count(DISTINCT (bank, account)) FROM account_window_features "
        f"WHERE {cond}"
    ).fetchone()
    lt = f"({_laundering_txns_sql(split_date)}) lt"
    n_launder = q(f"SELECT count(*) FROM {lt} WHERE {cond}")
    cited = q(
        f"SELECT count(*) FROM {lt} WHERE lt.{cond} AND txn_id IN "
        f"(SELECT unnest(rule_txn_ids) FROM {alerts_table} WHERE {cond})"
    )
    touched = q(
        f"""
        SELECT count(*) FROM {lt} WHERE lt.{cond} AND (
            EXISTS (SELECT 1 FROM {alerts_table} a WHERE a.bank = lt.from_bank
                    AND a.account = lt.from_account AND a.window_start = lt.window_start)
         OR EXISTS (SELECT 1 FROM {alerts_table} a WHERE a.bank = lt.to_bank
                    AND a.account = lt.to_account AND a.window_start = lt.window_start))
        """
    )
    return AlertMetrics(
        variant=variant,
        slice=slice_name,
        alerts=n_alerts,
        true_positives=tp,
        precision=tp / n_alerts if n_alerts else 0.0,
        positive_units=positives,
        recall=tp / positives if positives else 0.0,
        active_accounts=n_accounts,
        alerts_per_1k_accounts=1000 * n_alerts / n_accounts if n_accounts else 0.0,
        account_windows=n_windows,
        alerts_per_1k_windows=1000 * n_alerts / n_windows if n_windows else 0.0,
        laundering_txns=n_launder,
        txn_recall_cited=cited / n_launder if n_launder else 0.0,
        txn_recall_window=touched / n_launder if n_launder else 0.0,
    )


def typology_recall(
    conn: duckdb.DuckDBPyConnection, alerts_table: str, cond: str, split_date: date
) -> list[TypologyRecall]:
    rows = conn.execute(
        f"""
        WITH att AS (
            SELECT a.attempt_id, a.typology, t.from_bank, t.from_account, t.to_bank,
                   t.to_account, {window_start_sql("t.timestamp", split_date)} AS window_start
            FROM {EVAL_SCHEMA}.laundering_attempts a JOIN transactions t USING (txn_id)
        ), units AS (
            SELECT attempt_id, typology, from_bank AS bank, from_account AS account, window_start
            FROM att
            UNION
            SELECT attempt_id, typology, to_bank, to_account, window_start FROM att
        ), flagged AS (
            SELECT u.*, al.bank IS NOT NULL AS alerted
            FROM units u
            LEFT JOIN {alerts_table} al USING (bank, account, window_start)
            WHERE u.{cond}
        )
        SELECT typology,
               count(DISTINCT (bank, account, window_start)) AS units,
               count(DISTINCT (bank, account, window_start)) FILTER (alerted) AS units_alerted,
               count(DISTINCT attempt_id) AS attempts,
               count(DISTINCT attempt_id) FILTER (alerted) AS attempts_detected
        FROM flagged GROUP BY typology ORDER BY attempts DESC
        """
    ).fetchall()
    return [
        TypologyRecall(
            typology=t,
            units=u,
            units_alerted=ua,
            unit_recall=ua / u if u else 0.0,
            attempts=a,
            attempts_detected=ad,
            attempt_recall=ad / a if a else 0.0,
        )
        for t, u, ua, a, ad in rows
    ]


def rule_breakdown(
    conn: duckdb.DuckDBPyConnection, alerts_table: str, cond: str
) -> list[RuleBreakdown]:
    rows = conn.execute(
        f"""
        SELECT r.rule_id, count(*), count(p.bank)
        FROM (SELECT *, unnest(triggered_rules) AS rule_id FROM {alerts_table} WHERE {cond}) r
        LEFT JOIN {POSITIVE_UNITS} p USING (bank, account, window_start)
        GROUP BY 1 ORDER BY 2 DESC
        """
    ).fetchall()
    return [
        RuleBreakdown(rule_id=r, alerts=n, true_positives=tp, precision=tp / n) for r, n, tp in rows
    ]


def build_ablation_alerts(
    conn: duckdb.DuckDBPyConnection, score_col: str, budget_per_1k: float
) -> str:
    """Temp alerts table for one score column, cutoff set by budget on TRAIN scores."""
    train = conn.execute(
        f"SELECT {score_col} FROM window_scores WHERE split = 'train'"
    ).fetchnumpy()[score_col]
    tau = budget_threshold(train, budget_per_1k)
    name = f"_alerts_{score_col}"
    conn.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {alerts_select_sql(score_col, tau)}")
    return name
