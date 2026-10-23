from datetime import datetime, timedelta

import pytest

from aml.alerts.anomaly import fit_and_score
from aml.alerts.features import build_features
from aml.alerts.windows import build_legs
from aml.eval.alert_metrics import (
    build_positive_units,
    evaluate_slice,
    slice_conditions,
    typology_recall,
)
from aml.eval.tuning import tune_thresholds
from tests.alert_fixtures import SPLIT, Txn, fan_in, make_db, random_traffic


def _prepared(txns):
    conn = make_db(txns)
    build_legs(conn, SPLIT)
    build_features(conn)
    fit_and_score(conn)
    return conn


def _planted() -> list[Txn]:
    txns = random_traffic(n_accounts=150, n_txns=2500)
    for k in range(4):  # laundering fan-ins in train ...
        txns += fan_in(f"tr{k}_", f"HUB{k}", datetime(2022, 9, 2 + k, 8), 8)
    for k in range(2):  # ... and in test
        txns += fan_in(f"te{k}_", f"THUB{k}", datetime(2022, 9, 9 + k, 8), 8)
    return txns


# --- tuning --------------------------------------------------------------------


def test_tuning_meets_budget_on_train() -> None:
    conn = _prepared(_planted())
    result = tune_thresholds(conn, SPLIT, budget_per_1k=50)
    assert result.best.alerts <= int(result.train_windows * 50 / 1000)
    assert result.best.recall > 0


def test_tuning_ignores_test_labels() -> None:
    """Leakage guard: flipping every TEST label must not change the thresholds."""
    txns = _planted()
    a = tune_thresholds(_prepared(txns), SPLIT, budget_per_1k=50).thresholds
    flipped = [
        t if t.ts.date() < SPLIT else Txn(**{**t.__dict__, "laundering": not t.laundering})
        for t in txns
    ]
    b = tune_thresholds(_prepared(flipped), SPLIT, budget_per_1k=50).thresholds
    assert a == b


# --- metrics -------------------------------------------------------------------


@pytest.fixture
def scored_conn():
    """Hand-labelled world with a hand-written alerts table."""
    t = datetime(2022, 9, 9, 10, 0)
    txns = [
        Txn("L1", t, "A", "B", 100, laundering=True),  # positive units: A, B (test)
        Txn("L2", t + timedelta(hours=1), "B", "C", 100, laundering=True),  # adds C
        Txn("N1", t, "D", "E", 100),
        Txn("N2", datetime(2022, 9, 2), "A", "E", 100),  # train, clean
    ]
    conn = make_db(txns)
    conn.execute(
        "INSERT INTO eval.laundering_attempts VALUES (1, 'STACK', 'L1'), (1, 'STACK', 'L2')"
    )
    build_legs(conn, SPLIT)
    build_features(conn)
    build_positive_units(conn, SPLIT)
    conn.execute(
        """
        CREATE TABLE alerts AS SELECT * FROM (VALUES
            ('B', 'B', DATE '2022-09-08', ['L1'], ['FAN_IN']),   -- TP, cites L1
            ('B', 'D', DATE '2022-09-08', ['N1'], ['FAN_IN'])    -- FP
        ) v(bank, account, window_start, rule_txn_ids, triggered_rules)
        """
    )
    return conn


def test_precision_recall_and_txn_recall(scored_conn) -> None:
    cond = slice_conditions(SPLIT)["test"]
    m = evaluate_slice(scored_conn, "alerts", "combined", "test", cond, SPLIT)
    assert (m.alerts, m.true_positives, m.positive_units) == (2, 1, 3)
    assert m.precision == pytest.approx(0.5) and m.recall == pytest.approx(1 / 3)
    assert m.active_accounts == 5 and m.alerts_per_1k_accounts == pytest.approx(400)
    assert m.laundering_txns == 2
    assert m.txn_recall_cited == pytest.approx(0.5)  # only L1 is cited
    assert m.txn_recall_window == pytest.approx(1.0)  # B's window touches L1 and L2


def test_typology_recall(scored_conn) -> None:
    (t,) = typology_recall(scored_conn, "alerts", slice_conditions(SPLIT)["test"], SPLIT)
    assert (t.typology, t.units, t.units_alerted, t.attempts, t.attempts_detected) == (
        "STACK", 3, 1, 1, 1,
    )  # fmt: skip


def test_train_slice_has_no_positives(scored_conn) -> None:
    cond = slice_conditions(SPLIT)["train"]
    m = evaluate_slice(scored_conn, "alerts", "combined", "train", cond, SPLIT)
    assert (m.alerts, m.positive_units, m.recall) == (0, 0, 0.0)
