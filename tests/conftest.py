from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pytest

from aml.data.customers import build_customers
from aml.data.ingest import build_transactions, load_raw_lines
from aml.data.split import assign_split
from aml.eval.labels import build_txn_labels
from aml.eval.patterns import PatternMatchReport, build_laundering_attempts, parse_patterns

HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)

# Each row exercises one case seen in HI-Small.
ROWS = [
    "2022/09/01 00:20,010,8000EBD30,010,8000EBD30,36.97,US Dollar,36.97,US Dollar,Reinvestment,0",
    # exact duplicates (rows 1 and 2)
    "2022/09/01 00:21,03208,8000F4580,001,8000F5340,0.01,US Dollar,0.01,US Dollar,Cheque,0",
    "2022/09/01 00:21,03208,8000F4580,001,8000F5340,0.01,US Dollar,0.01,US Dollar,Cheque,0",
    # cross-currency: 100 EUR paid, USD received at the table rate
    "2022/09/02 10:00,0220,800A00001,0330,800B00002,117.18,US Dollar,100.00,Euro,Wire,0",
    # laundering rows that belong to the FAN-OUT attempt in PATTERNS
    "2022/09/03 09:29,021174,800737690,001,800AAF0B0,15197.45,US Dollar,15197.45,US Dollar,ACH,1",
    "2022/09/04 07:35,021174,800737690,010,800449990,0.038417,Bitcoin,0.038417,Bitcoin,Bitcoin,1",
    # laundering row not in any attempt
    "2022/09/05 12:00,0440,800C00003,0550,800D00004,5000.00,Yen,5000.00,Yen,Cash,1",
    # same account ID under a different bank than row 0's from_account
    "2022/09/06 08:00,999,8000EBD30,010,8000EBD30,10.00,US Dollar,10.00,US Dollar,ACH,0",
    # split boundary: last train minute, first test minute
    "2022/09/07 23:59,010,8000EBD30,0220,800A00001,1.00,UK Pound,1.00,UK Pound,ACH,0",
    "2022/09/08 00:00,010,8000EBD30,0220,800A00001,2.00,UK Pound,2.00,UK Pound,ACH,0",
]

PATTERNS = (
    "BEGIN LAUNDERING ATTEMPT - FAN-OUT: Max 2-degree Fan-Out\r\n"
    f"{ROWS[4]}\r\n"
    f"{ROWS[5]}\r\n"
    "END LAUNDERING ATTEMPT - FAN-OUT\r\n"
    "\r\n"
    "BEGIN LAUNDERING ATTEMPT - CYCLE: Max 1 hops\r\n"
    "2022/09/09 00:00,1,NOT_IN_CSV,2,X,1.00,US Dollar,1.00,US Dollar,ACH,1\r\n"
    "END LAUNDERING ATTEMPT - CYCLE\r\n"
)

SPLIT_DATE = date(2022, 9, 8)


@pytest.fixture
def raw_files(tmp_path: Path) -> tuple[Path, Path]:
    csv_path = tmp_path / "trans.csv"
    csv_path.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    patterns_path = tmp_path / "patterns.txt"
    patterns_path.write_bytes(PATTERNS.encode())
    return csv_path, patterns_path


@dataclass
class Ingested:
    conn: duckdb.DuckDBPyConnection
    patterns: PatternMatchReport


@pytest.fixture
def ingested(raw_files: tuple[Path, Path]) -> Iterator[Ingested]:
    """In-memory DB after the full ingest pipeline has run on the fixture."""
    csv_path, patterns_path = raw_files
    c = duckdb.connect()
    load_raw_lines(c, csv_path)
    build_transactions(c)
    assign_split(c, SPLIT_DATE)
    build_txn_labels(c)
    report = build_laundering_attempts(c, parse_patterns(patterns_path))
    build_customers(c, seed=42)
    yield Ingested(c, report)
    c.close()


@pytest.fixture
def conn(ingested: Ingested) -> duckdb.DuckDBPyConnection:
    return ingested.conn
