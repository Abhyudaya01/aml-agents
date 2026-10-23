"""Per (account, window) features, computed from that window's legs only.

Nothing here looks outside the window: no lifetime degree, no "first seen" date,
no global statistics. That keeps features leak-free across the temporal split and
makes each feature explainable from the alert's own transactions.
"""

import duckdb

# Domain constant (US CTR threshold), not fit to data.
STRUCTURING_THRESHOLD_USD = 10_000
STRUCTURING_BAND = 0.10  # "within 10% below" the threshold
ROUND_AMOUNT_MIN = 1_000
ROUND_AMOUNT_UNIT = 100

PAYMENT_FORMATS = ("ACH", "Cheque", "Credit Card", "Cash", "Wire", "Bitcoin", "Reinvestment")

# Pass-through: an in-leg followed by an out-leg at a similar amount.
PASS_THROUGH_MAX_HOURS = 48
PASS_THROUGH_AMOUNT_TOL = 0.10


def _format_col(fmt: str) -> str:
    return "share_" + fmt.lower().replace(" ", "_")


# Columns fed to the anomaly model (all numeric, all from the spec's feature list).
MODEL_FEATURES: tuple[str, ...] = (
    "n_txn",
    "n_in",
    "n_out",
    "usd_in",
    "usd_out",
    "cp_in",
    "cp_out",
    "cross_currency_share",
    *(_format_col(f) for f in PAYMENT_FORMATS),
    "max_24h",
    "burst_ratio",
    "n_near_threshold",
    "round_share",
)


def build_features(conn: duckdb.DuckDBPyConnection) -> int:
    low = STRUCTURING_THRESHOLD_USD * (1 - STRUCTURING_BAND)
    format_shares = ",\n               ".join(
        f"avg(CAST(payment_format = '{f}' AS INTEGER)) AS {_format_col(f)}" for f in PAYMENT_FORMATS
    )
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE account_window_features AS
        WITH rolled AS (
            SELECT *, cp_bank || ':' || cp_account AS cp,
                   count(*) OVER (
                       PARTITION BY bank, account, window_start ORDER BY ts
                       RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW
                   ) AS n_24h
            FROM legs
        )
        SELECT bank, account, window_start, any_value(split) AS split,
               count(*) AS n_txn,
               count(*) FILTER (direction = 'in') AS n_in,
               count(*) FILTER (direction = 'out') AS n_out,
               CAST(coalesce(sum(amount_usd) FILTER (direction = 'in'), 0) AS DOUBLE) AS usd_in,
               CAST(coalesce(sum(amount_usd) FILTER (direction = 'out'), 0) AS DOUBLE) AS usd_out,
               count(DISTINCT cp) FILTER (direction = 'in') AS cp_in,
               count(DISTINCT cp) FILTER (direction = 'out') AS cp_out,
               avg(CAST(cross_currency AS INTEGER)) AS cross_currency_share,
               {format_shares},
               max(n_24h) AS max_24h,
               max(n_24h) / (count(*) / {7.0}) AS burst_ratio,
               arg_max(ts, n_24h) AS peak_24h_end,
               count(*) FILTER (amount_usd >= {low} AND amount_usd < {STRUCTURING_THRESHOLD_USD})
                   AS n_near_threshold,
               avg(CAST(amount >= {ROUND_AMOUNT_MIN} AND amount % {ROUND_AMOUNT_UNIT} = 0
                        AS INTEGER)) AS round_share
        FROM rolled
        GROUP BY bank, account, window_start
        """
    )
    _add_pass_through(conn)
    return conn.execute("SELECT count(*) FROM account_window_features").fetchone()[0]


def _add_pass_through(conn: duckdb.DuckDBPyConnection) -> None:
    """Pairs (in-leg, out-leg) on the same account and window where the out-leg
    follows within 48h at an amount within ±10%, to a different counterparty.

    Stored per window as the number of in-legs that were passed through, plus the
    txn_ids involved (used as rule evidence).
    """
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE pass_through AS
        WITH pairs AS (
            SELECT i.bank, i.account, i.window_start, i.txn_id AS in_txn, o.txn_id AS out_txn
            FROM legs i JOIN legs o
              ON o.bank = i.bank AND o.account = i.account AND o.window_start = i.window_start
             AND o.ts >= i.ts AND o.ts <= i.ts + INTERVAL {PASS_THROUGH_MAX_HOURS} HOURS
             AND o.amount_usd >= i.amount_usd * {1 - PASS_THROUGH_AMOUNT_TOL}
             AND o.amount_usd <= i.amount_usd * {1 + PASS_THROUGH_AMOUNT_TOL}
            WHERE i.direction = 'in' AND o.direction = 'out' AND i.amount_usd > 0
              AND NOT (o.cp_bank = i.cp_bank AND o.cp_account = i.cp_account)
        )
        SELECT bank, account, window_start,
               count(DISTINCT in_txn) AS n_pass_through,
               list_distinct(list(in_txn) || list(out_txn)) AS pass_through_txn_ids
        FROM pairs GROUP BY ALL
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE account_window_features AS
        SELECT f.*, coalesce(p.n_pass_through, 0) AS n_pass_through
        FROM account_window_features f
        LEFT JOIN pass_through p USING (bank, account, window_start)
        """
    )
