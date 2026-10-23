"""Stable transaction IDs.

Why this exists: the source CSV has no ID column, yet every downstream artifact
(alerts, evidence, narrative claims, the citation validator, the audit log) needs
to point at a specific transaction. We therefore mint one at ingest.

    txn_id = first 16 hex chars of sha256(raw_line + "|" + row_index)

- `raw_line` is the exact source line (without the line terminator), so the ID is
  tied to the row's content rather than to a mutable position alone.
- `row_index` is the 0-based position of the data line in the file (header
  excluded). It breaks ties: the source contains exact duplicate rows (9 groups,
  18 rows in HI-Small), which would otherwise hash to the same ID.
- 16 hex chars = 64 bits; for ~5M rows the collision probability is ~1e-6, and
  ingest asserts uniqueness anyway.

IDs are reproducible as long as the source file is unchanged.
"""

import hashlib

TXN_ID_HEX_CHARS = 16


def mint_txn_id(raw_line: str, row_index: int) -> str:
    """Python reference implementation; must agree with `txn_id_sql`."""
    digest = hashlib.sha256(f"{raw_line}|{row_index}".encode()).hexdigest()
    return digest[:TXN_ID_HEX_CHARS]


def txn_id_sql(line_col: str = "line", index_col: str = "row_index") -> str:
    """DuckDB expression computing the same ID inside the database."""
    return (
        f"substr(sha256({line_col} || '|' || CAST({index_col} AS VARCHAR)), 1, {TXN_ID_HEX_CHARS})"
    )
