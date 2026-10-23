"""Load the raw transactions CSV into DuckDB.

Two stages:
1. `load_raw_lines` reads each data line verbatim into a temp table with its
   0-based `row_index`. Keeping the exact text is what makes txn_ids reproducible.
2. `build_transactions` splits lines into typed, snake_case columns, mints txn_ids,
   and adds USD amounts. Only the first 10 fields are parsed; the target column
   is left for the evaluation package.
"""

from pathlib import Path

import duckdb

from aml.data.fx import load_fx_table, unknown_currencies
from aml.data.schema import SOURCE_COLUMNS, SOURCE_FIELD_COUNT, SOURCE_TIMESTAMP_FORMAT
from aml.data.txn_id import txn_id_sql

RAW_TABLE = "raw_lines"


def _read_header(csv_path: Path) -> list[str]:
    with csv_path.open() as f:
        return f.readline().rstrip("\r\n").split(",")


def validate_header(csv_path: Path) -> None:
    header = _read_header(csv_path)
    expected = [name for _, name, _ in SOURCE_COLUMNS]
    if len(header) != SOURCE_FIELD_COUNT or header[: len(expected)] != expected:
        raise ValueError(f"Unexpected header in {csv_path}: {header}")


def load_raw_lines(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> int:
    """Create temp table raw_lines(row_index, line). Returns the row count."""
    validate_header(csv_path)
    conn.execute("SET preserve_insertion_order = true")
    # Read every line as a single VARCHAR (the delimiter never occurs), so the
    # text is untouched. rowid follows insertion order, i.e. file order.
    conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE _raw_read AS
        SELECT line FROM read_csv(?, columns = {'line': 'VARCHAR'}, header = true,
                                  delim = '\x1f', quote = '', escape = '', auto_detect = false)
        """,
        [str(csv_path)],
    )
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {RAW_TABLE} AS
        SELECT rowid AS row_index, rtrim(line, chr(13)) AS line FROM _raw_read ORDER BY rowid
        """
    )
    conn.execute("DROP TABLE _raw_read")
    bad = conn.execute(
        f"SELECT count(*) FROM {RAW_TABLE} WHERE len(string_split(line, ',')) <> ?",
        [SOURCE_FIELD_COUNT],
    ).fetchone()[0]
    if bad:
        raise ValueError(f"{bad} lines do not have {SOURCE_FIELD_COUNT} fields")
    return conn.execute(f"SELECT count(*) FROM {RAW_TABLE}").fetchone()[0]


def check_row_order(conn: duckdb.DuckDBPyConnection, csv_path: Path, every: int = 100_000) -> int:
    """Spot-check that raw_lines.row_index matches the line's position in the file.

    Compares every `every`-th line (plus the first and last) read independently in
    Python. Returns the number of lines checked; raises on any mismatch.
    """
    expected: dict[int, str] = {}
    with csv_path.open() as f:
        next(f)  # header
        last = (-1, "")
        for i, line in enumerate(f):
            last = (i, line.rstrip("\r\n"))
            if i % every == 0:
                expected[i] = last[1]
    expected[last[0]] = last[1]
    got = dict(
        conn.execute(
            f"SELECT row_index, line FROM {RAW_TABLE} WHERE row_index IN "
            f"(SELECT unnest(?::BIGINT[]))",
            [list(expected)],
        ).fetchall()
    )
    if got != expected:
        bad = sorted(k for k in expected if got.get(k) != expected[k])
        raise ValueError(f"raw_lines order differs from file at row_index {bad[:5]}")
    return len(expected)


def _field_select() -> str:
    casts = {
        "timestamp": f"strptime(f[{{i}}], '{SOURCE_TIMESTAMP_FORMAT}')",
        "amount_received": "CAST(f[{i}] AS DECIMAL(38, 6))",
        "amount_paid": "CAST(f[{i}] AS DECIMAL(38, 6))",
    }
    parts = [casts.get(col, "f[{i}]").format(i=i) + f" AS {col}" for i, _, col in SOURCE_COLUMNS]
    return ",\n            ".join(parts)


def build_transactions(conn: duckdb.DuckDBPyConnection) -> int:
    """Create `transactions` from raw_lines. Returns the row count."""
    load_fx_table(conn)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE transactions AS
        WITH parsed AS (
            SELECT {txn_id_sql()} AS txn_id, row_index, string_split(line, ',') AS f
            FROM {RAW_TABLE}
        ), typed AS (
            SELECT txn_id, row_index,
            {_field_select()}
            FROM parsed
        )
        SELECT t.*,
            CAST(round(t.amount_received * r.usd_per_unit, 2) AS DECIMAL(38, 2))
                AS amount_received_usd,
            CAST(round(t.amount_paid * p.usd_per_unit, 2) AS DECIMAL(38, 2))
                AS amount_paid_usd,
            CAST(NULL AS VARCHAR) AS split
        FROM typed t
        LEFT JOIN fx_rates r ON r.currency = t.receiving_currency
        LEFT JOIN fx_rates p ON p.currency = t.payment_currency
        ORDER BY t.row_index
        """
    )
    missing = unknown_currencies(conn)
    if missing:
        raise ValueError(f"No FX rate for currencies: {missing}")
    n, distinct = conn.execute(
        "SELECT count(*), count(DISTINCT txn_id) FROM transactions"
    ).fetchone()
    if n != distinct:
        raise ValueError(f"txn_id collision: {n} rows, {distinct} distinct ids")
    conn.execute("ALTER TABLE transactions ADD PRIMARY KEY (txn_id)")
    return n


def drop_raw_lines(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {RAW_TABLE}")
