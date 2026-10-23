"""Tiny hand-built `transactions` tables for alerting tests (no ingest needed)."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import duckdb
import numpy as np

SPLIT = date(2022, 9, 8)
T0 = datetime(2022, 9, 1, 0, 0)


@dataclass
class Txn:
    txn_id: str
    ts: datetime
    src: str  # account id; bank is fixed to "B" unless given as "bank:acct"
    dst: str
    usd: float
    fmt: str = "ACH"
    cross_currency: bool = False
    laundering: bool = False


def _split_key(key: str) -> tuple[str, str]:
    return tuple(key.split(":", 1)) if ":" in key else ("B", key)  # type: ignore[return-value]


def make_db(txns: list[Txn]) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute(
        """
        CREATE TABLE transactions (
            txn_id VARCHAR, row_index BIGINT, timestamp TIMESTAMP,
            from_bank VARCHAR, from_account VARCHAR, to_bank VARCHAR, to_account VARCHAR,
            amount_received DECIMAL(38, 6), receiving_currency VARCHAR,
            amount_paid DECIMAL(38, 6), payment_currency VARCHAR, payment_format VARCHAR,
            amount_received_usd DECIMAL(38, 2), amount_paid_usd DECIMAL(38, 2), split VARCHAR
        )
        """
    )
    rows = []
    for i, t in enumerate(txns):
        fb, fa = _split_key(t.src)
        tb, ta = _split_key(t.dst)
        pay_ccy = "Euro" if t.cross_currency else "US Dollar"
        split = "train" if t.ts.date() < SPLIT else "test"
        rows.append(
            (t.txn_id, i, t.ts, fb, fa, tb, ta, t.usd, "US Dollar", t.usd, pay_ccy, t.fmt,
             t.usd, t.usd, split)
        )  # fmt: skip
    conn.executemany("INSERT INTO transactions VALUES (" + ", ".join("?" * 15) + ")", rows)
    conn.execute("CREATE SCHEMA eval")
    conn.execute("CREATE TABLE eval.txn_labels (txn_id VARCHAR, is_laundering BOOLEAN)")
    conn.executemany(
        "INSERT INTO eval.txn_labels VALUES (?, ?)", [(t.txn_id, t.laundering) for t in txns]
    )
    conn.execute(
        "CREATE TABLE eval.laundering_attempts (attempt_id INTEGER, typology VARCHAR, "
        "txn_id VARCHAR)"
    )
    return conn


def random_traffic(n_accounts: int = 120, n_txns: int = 1500, seed: int = 0) -> list[Txn]:
    """Background noise over Sep 1-14: random pairs, log-normal amounts."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_txns):
        a, b = rng.choice(n_accounts, size=2, replace=False)
        ts = T0 + timedelta(minutes=int(rng.integers(0, 14 * 24 * 60)))
        out.append(Txn(f"n{i}", ts, f"A{a}", f"A{b}", round(float(rng.lognormal(6, 1)), 2)))
    return out


def fan_in(prefix: str, hub: str, start: datetime, n: int, laundering: bool = True) -> list[Txn]:
    """n distinct senders pay `hub` within a few hours."""
    return [
        Txn(f"{prefix}{i}", start + timedelta(minutes=10 * i), f"{prefix}S{i}", hub, 4000.0,
            laundering=laundering)
        for i in range(n)
    ]  # fmt: skip
