"""Account "legs" and 7-day tumbling windows.

A transaction A -> B becomes two legs: an `out` leg for A and an `in` leg for B.
The alert unit is (bank, account, window_start).

Windows are tumbling 7-day blocks anchored on the train/test split date, so a
window boundary always coincides with the split boundary: no window mixes train
and test transactions. `build_legs` asserts this.

Self-transfers (same bank and account on both sides; 591k "Reinvestment" rows in
HI-Small) carry no counterparty information and are excluded from legs.
"""

from datetime import date

import duckdb

WINDOW_DAYS = 7


def window_start_sql(ts_col: str, split_date: date) -> str:
    """DuckDB expression mapping a timestamp to its window's first day."""
    anchor = f"DATE '{split_date.isoformat()}'"
    return (
        f"{anchor} + {WINDOW_DAYS} * CAST(floor(date_diff('day', {anchor}, "
        f"CAST({ts_col} AS DATE)) / {WINDOW_DAYS}.0) AS INTEGER)"
    )


def build_legs(conn: duckdb.DuckDBPyConnection, split_date: date) -> int:
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE legs AS
        WITH base AS (
            SELECT *, receiving_currency <> payment_currency AS cross_currency,
                   {window_start_sql("timestamp", split_date)} AS window_start
            FROM transactions
            WHERE NOT (from_bank = to_bank AND from_account = to_account)
        )
        SELECT txn_id, from_bank AS bank, from_account AS account, 'out' AS direction,
               to_bank AS cp_bank, to_account AS cp_account, timestamp AS ts,
               amount_paid_usd AS amount_usd, amount_paid AS amount,
               payment_format, cross_currency, window_start, split
        FROM base
        UNION ALL
        SELECT txn_id, to_bank, to_account, 'in',
               from_bank, from_account, timestamp,
               amount_received_usd, amount_received,
               payment_format, cross_currency, window_start, split
        FROM base
        """
    )
    straddling = conn.execute(
        "SELECT count(*) FROM (SELECT window_start FROM legs GROUP BY 1 "
        "HAVING count(DISTINCT split) > 1)"
    ).fetchone()[0]
    if straddling:
        raise ValueError(
            f"{straddling} windows mix train and test rows; the split date used at "
            "ingest must equal the one used for windows"
        )
    return conn.execute("SELECT count(*) FROM legs").fetchone()[0]
