"""Temporal train/test split.

Rows strictly before `split_date` (midnight) are train; the rest are test. See
docs/PROGRESS.md (Phase 1) for why the default date is 2022-09-08.
"""

from datetime import date

import duckdb


def assign_split(conn: duckdb.DuckDBPyConnection, split_date: date) -> dict[str, int]:
    conn.execute(
        "UPDATE transactions SET split = CASE WHEN timestamp < ? THEN 'train' ELSE 'test' END",
        [split_date],
    )
    rows = conn.execute("SELECT split, count(*) FROM transactions GROUP BY 1").fetchall()
    return dict(rows)
