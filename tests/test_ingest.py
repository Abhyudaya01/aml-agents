from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from aml.data.customers import generate_customers
from aml.data.fx import USD_PER_UNIT, fx_consistency
from aml.data.ingest import build_transactions, check_row_order, load_raw_lines
from aml.data.txn_id import mint_txn_id
from aml.eval.labels import laundering_rate_by_split
from aml.eval.patterns import parse_patterns
from tests.conftest import ROWS


def test_columns_are_snake_case_with_from_and_to_account(conn) -> None:
    cols = [r[0] for r in conn.execute("DESCRIBE transactions").fetchall()]
    assert cols == [
        "txn_id", "row_index", "timestamp", "from_bank", "from_account", "to_bank",
        "to_account", "amount_received", "receiving_currency", "amount_paid",
        "payment_currency", "payment_format", "amount_received_usd", "amount_paid_usd", "split",
    ]  # fmt: skip
    row = conn.execute(
        "SELECT from_bank, from_account, to_bank, to_account FROM transactions WHERE row_index = 1"
    ).fetchone()
    assert row == ("03208", "8000F4580", "001", "8000F5340")  # leading zeros kept


def test_transactions_table_has_no_label_column(conn) -> None:
    cols = {r[0] for r in conn.execute("DESCRIBE transactions").fetchall()}
    assert not any("launder" in c for c in cols)


def test_txn_id_matches_reference_and_is_unique(conn) -> None:
    ids = dict(conn.execute("SELECT row_index, txn_id FROM transactions").fetchall())
    assert ids == {i: mint_txn_id(line, i) for i, line in enumerate(ROWS)}
    assert ids[1] != ids[2]  # exact duplicate rows still get distinct ids


def test_txn_id_is_stable_across_runs(raw_files: tuple[Path, Path]) -> None:
    def ids() -> list[str]:
        c = duckdb.connect()
        load_raw_lines(c, raw_files[0])
        build_transactions(c)
        rows = c.execute("SELECT txn_id FROM transactions ORDER BY row_index").fetchall()
        return [r[0] for r in rows]

    assert ids() == ids()


def test_row_order_check_passes(raw_files: tuple[Path, Path]) -> None:
    c = duckdb.connect()
    load_raw_lines(c, raw_files[0])
    assert check_row_order(c, raw_files[0], every=3) >= 4


def test_header_validation_rejects_unexpected_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="Unexpected header"):
        load_raw_lines(duckdb.connect(), bad)


def test_usd_amounts(conn) -> None:
    amounts = conn.execute(
        "SELECT amount_paid, amount_paid_usd, amount_received_usd FROM transactions "
        "WHERE row_index = 3"
    ).fetchone()
    assert amounts == (Decimal("100.000000"), Decimal("117.18"), Decimal("117.18"))
    btc = conn.execute("SELECT amount_paid_usd FROM transactions WHERE row_index = 5").fetchone()
    assert btc[0] == Decimal(str(round(0.038417 * USD_PER_UNIT["Bitcoin"], 2)))


def test_fx_consistency_on_cross_currency_rows(conn) -> None:
    fx = fx_consistency(conn)
    assert fx["rows"] == 1 and fx["median_rel_err"] < 1e-3


def test_unknown_currency_fails(tmp_path: Path) -> None:
    from tests.conftest import HEADER

    p = tmp_path / "t.csv"
    p.write_text(HEADER + "\n2022/09/01 00:00,1,A,2,B,1.0,Doubloon,1.0,Doubloon,ACH,0\n")
    c = duckdb.connect()
    load_raw_lines(c, p)
    with pytest.raises(ValueError, match="Doubloon"):
        build_transactions(c)


def test_split_boundary(conn) -> None:
    splits = dict(conn.execute("SELECT row_index, split FROM transactions").fetchall())
    assert splits[8] == "train" and splits[9] == "test"
    assert set(splits.values()) == {"train", "test"}


def test_laundering_rate_by_split(conn) -> None:
    rates = laundering_rate_by_split(conn)
    assert rates["train"]["laundering"] == 3 and rates["test"]["laundering"] == 0


def test_parse_patterns_handles_crlf_and_detail(raw_files: tuple[Path, Path]) -> None:
    attempts = parse_patterns(raw_files[1])
    assert [(a.typology, a.detail, len(a.lines)) for a in attempts] == [
        ("FAN-OUT", "Max 2-degree Fan-Out", 2),
        ("CYCLE", "Max 1 hops", 1),
    ]
    assert attempts[0].lines[0] == ROWS[4]  # no trailing \r


def test_parse_patterns_rejects_unclosed_block(tmp_path: Path) -> None:
    p = tmp_path / "p.txt"
    p.write_text("BEGIN LAUNDERING ATTEMPT - STACK\n" + ROWS[4] + "\n")
    with pytest.raises(ValueError, match="open attempt"):
        parse_patterns(p)


def test_laundering_attempts_match(ingested) -> None:
    conn = ingested.conn
    rows = conn.execute(
        "SELECT a.attempt_id, a.typology, t.row_index FROM eval.laundering_attempts a "
        "JOIN transactions t USING (txn_id) ORDER BY 1, 3"
    ).fetchall()
    assert rows == [(1, "FAN-OUT", 4), (1, "FAN-OUT", 5)]
    report = ingested.patterns
    assert (report.matched_rows, report.pattern_rows, report.fully_matched_attempts) == (2, 3, 1)
    assert report.positives_covered == 2 and report.labeled_positive_rows == 3
    assert report.matched_but_unlabeled == 0


def test_customers_one_per_bank_account(conn) -> None:
    n, distinct = conn.execute(
        "SELECT count(*), count(DISTINCT (bank, account)) FROM customers"
    ).fetchone()
    accounts = {(r.split(",")[1], r.split(",")[2]) for r in ROWS} | {
        (r.split(",")[3], r.split(",")[4]) for r in ROWS
    }
    assert n == distinct == len(accounts)
    # 8000EBD30 exists under two banks -> two customers
    dual = conn.execute("SELECT count(*) FROM customers WHERE account = '8000EBD30'").fetchone()
    assert dual[0] == 2


def test_customers_are_deterministic() -> None:
    accounts = [("010", "A"), ("001", "B"), ("010", "A")]
    first, second = generate_customers(accounts, seed=7), generate_customers(accounts, seed=7)
    assert first == second and len(first) == 2
    assert generate_customers(accounts, seed=8) != first
    assert all(c.name and c.address and c.occupation and "\n" not in c.address for c in first)
