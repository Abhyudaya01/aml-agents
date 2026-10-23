"""Ground-truth labels. Only `aml.eval` may create or read these tables.

They live in a separate DuckDB schema, `eval`, so that the boundary is visible in
the database as well as in the code.
"""

import duckdb

from aml.data.ingest import RAW_TABLE
from aml.data.schema import SOURCE_FIELD_COUNT

EVAL_SCHEMA = "eval"


def build_txn_labels(conn: duckdb.DuckDBPyConnection) -> int:
    """Create eval.txn_labels(txn_id, is_laundering) from the raw lines' last field."""
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {EVAL_SCHEMA}")
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {EVAL_SCHEMA}.txn_labels AS
        SELECT t.txn_id,
               CAST(string_split(r.line, ',')[{SOURCE_FIELD_COUNT}] AS INTEGER) = 1 AS is_laundering
        FROM {RAW_TABLE} r JOIN transactions t USING (row_index)
        """
    )
    return conn.execute(f"SELECT count(*) FROM {EVAL_SCHEMA}.txn_labels").fetchone()[0]


def laundering_rate_by_split(conn: duckdb.DuckDBPyConnection) -> dict[str, dict[str, float]]:
    rows = conn.execute(
        f"""
        SELECT t.split, count(*), sum(CAST(l.is_laundering AS INTEGER))
        FROM transactions t JOIN {EVAL_SCHEMA}.txn_labels l USING (txn_id)
        GROUP BY 1 ORDER BY 1 DESC
        """
    ).fetchall()
    return {split: {"rows": n, "laundering": int(pos), "rate": pos / n} for split, n, pos in rows}
