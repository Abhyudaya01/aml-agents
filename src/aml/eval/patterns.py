"""Parse the HI-Small patterns file and link its rows to minted txn_ids.

File format (CRLF line endings), blocks separated by blank lines:

    BEGIN LAUNDERING ATTEMPT - FAN-OUT: Max 16-degree Fan-Out
    2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,...,ACH,1
    ...
    END LAUNDERING ATTEMPT - FAN-OUT

Rows use the exact transactions-CSV format, so they are matched to transactions by
their raw line text. If a line occurs k times in the source (exact duplicates),
the n-th occurrence in the patterns file is paired with the n-th occurrence in
file order, so each transaction is assigned at most once.
"""

import re
from pathlib import Path

import duckdb
from pydantic import BaseModel

from aml.data.ingest import RAW_TABLE
from aml.eval.labels import EVAL_SCHEMA

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - (?P<typology>[A-Z-]+)(?::\s*(?P<detail>.*))?$")
_END = re.compile(r"^END LAUNDERING ATTEMPT - (?P<typology>[A-Z-]+)$")


class PatternAttempt(BaseModel):
    attempt_id: int
    typology: str
    detail: str | None
    lines: list[str]


class PatternMatchReport(BaseModel):
    attempts: int
    pattern_rows: int
    matched_rows: int
    fully_matched_attempts: int
    labeled_positive_rows: int
    positives_covered: int
    matched_but_unlabeled: int
    attempts_by_typology: dict[str, int]
    rows_by_typology: dict[str, int]

    @property
    def match_rate(self) -> float:
        return self.matched_rows / self.pattern_rows if self.pattern_rows else 0.0


def parse_patterns(path: Path) -> list[PatternAttempt]:
    attempts: list[PatternAttempt] = []
    current: PatternAttempt | None = None
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if m := _BEGIN.match(line):
            if current is not None:
                raise ValueError(f"line {lineno}: BEGIN inside an open attempt")
            current = PatternAttempt(
                attempt_id=len(attempts) + 1,
                typology=m["typology"],
                detail=m["detail"],
                lines=[],
            )
        elif m := _END.match(line):
            if current is None or m["typology"] != current.typology:
                raise ValueError(f"line {lineno}: unmatched END")
            attempts.append(current)
            current = None
        elif current is not None:
            current.lines.append(line)
        else:
            raise ValueError(f"line {lineno}: transaction row outside an attempt")
    if current is not None:
        raise ValueError("file ends inside an open attempt")
    return attempts


def build_laundering_attempts(
    conn: duckdb.DuckDBPyConnection, attempts: list[PatternAttempt]
) -> PatternMatchReport:
    """Create eval.laundering_attempts(attempt_id, typology, txn_id)."""
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {EVAL_SCHEMA}")
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE _pattern_lines "
        "(attempt_id INTEGER, typology VARCHAR, seq INTEGER, line VARCHAR)"
    )
    conn.executemany(
        "INSERT INTO _pattern_lines VALUES (?, ?, ?, ?)",
        [(a.attempt_id, a.typology, i, ln) for a in attempts for i, ln in enumerate(a.lines)],
    )
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE {EVAL_SCHEMA}.laundering_attempts AS
        WITH p AS (
            SELECT *, row_number() OVER (PARTITION BY line ORDER BY attempt_id, seq) AS occ
            FROM _pattern_lines
        ), r AS (
            SELECT row_index, line,
                   row_number() OVER (PARTITION BY line ORDER BY row_index) AS occ
            FROM {RAW_TABLE}
            WHERE line IN (SELECT line FROM _pattern_lines)
        )
        SELECT p.attempt_id, p.typology, t.txn_id
        FROM p
        JOIN r USING (line, occ)
        JOIN transactions t USING (row_index)
        ORDER BY p.attempt_id, p.seq
        """
    )
    report = _match_report(conn, attempts)
    conn.execute("DROP TABLE _pattern_lines")
    return report


def _match_report(
    conn: duckdb.DuckDBPyConnection, attempts: list[PatternAttempt]
) -> PatternMatchReport:
    s = EVAL_SCHEMA
    matched = conn.execute(f"SELECT count(*) FROM {s}.laundering_attempts").fetchone()[0]
    fully = conn.execute(
        f"""
        SELECT count(*) FROM (SELECT attempt_id, count(*) AS n FROM {s}.laundering_attempts
                              GROUP BY 1) m
        JOIN (SELECT attempt_id, count(*) AS n FROM _pattern_lines GROUP BY 1) p
        USING (attempt_id) WHERE m.n = p.n
        """
    ).fetchone()[0]
    positives, covered, unlabeled = conn.execute(
        f"""
        SELECT
            (SELECT count(*) FROM {s}.txn_labels WHERE is_laundering),
            (SELECT count(DISTINCT a.txn_id) FROM {s}.laundering_attempts a
             JOIN {s}.txn_labels l USING (txn_id) WHERE l.is_laundering),
            (SELECT count(DISTINCT a.txn_id) FROM {s}.laundering_attempts a
             JOIN {s}.txn_labels l USING (txn_id) WHERE NOT l.is_laundering)
        """
    ).fetchone()
    by_typ_attempts: dict[str, int] = {}
    by_typ_rows: dict[str, int] = {}
    for a in attempts:
        by_typ_attempts[a.typology] = by_typ_attempts.get(a.typology, 0) + 1
        by_typ_rows[a.typology] = by_typ_rows.get(a.typology, 0) + len(a.lines)
    return PatternMatchReport(
        attempts=len(attempts),
        pattern_rows=sum(len(a.lines) for a in attempts),
        matched_rows=matched,
        fully_matched_attempts=fully,
        labeled_positive_rows=positives,
        positives_covered=covered,
        matched_but_unlabeled=unlabeled,
        attempts_by_typology=dict(sorted(by_typ_attempts.items(), key=lambda kv: -kv[1])),
        rows_by_typology=dict(sorted(by_typ_rows.items(), key=lambda kv: -kv[1])),
    )
