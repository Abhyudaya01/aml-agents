from datetime import date, datetime, timedelta

import numpy as np
import pytest

from aml.alerts.anomaly import fit_and_score
from aml.alerts.features import build_features
from aml.alerts.records import (
    ANOMALY_WEIGHT,
    RULE_WEIGHT,
    budget_threshold,
    build_alerts,
    fetch_alert,
    score_windows,
)
from aml.alerts.rules import build_rule_hits
from aml.alerts.thresholds import RuleThresholds, load_thresholds, save_thresholds
from aml.alerts.windows import build_legs
from tests.alert_fixtures import SPLIT, Txn, fan_in, make_db, random_traffic

ALL_ON = RuleThresholds(
    fan_in_min_cp=3,
    fan_out_min_cp=3,
    structuring_min_count=2,
    velocity_min_24h=4,
    velocity_min_burst=5,
    pass_through_min_count=1,
    alert_threshold=0.3,
    budget_per_1k_windows=10,
)


def _features(txns):
    conn = make_db(txns)
    build_legs(conn, SPLIT)
    build_features(conn)
    return conn


def _row(conn, account, window=date(2022, 9, 1)):
    cur = conn.execute(
        "SELECT * FROM account_window_features WHERE account = ? AND window_start = ?",
        [account, window],
    )
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, cur.fetchone(), strict=True))


# --- windows -------------------------------------------------------------------


def test_windows_align_with_split_and_drop_self_transfers() -> None:
    conn = make_db(
        [
            Txn("a", datetime(2022, 9, 1, 0, 0), "X", "Y", 1),
            Txn("b", datetime(2022, 9, 7, 23, 59), "X", "Y", 1),
            Txn("c", datetime(2022, 9, 8, 0, 0), "X", "Y", 1),
            Txn("d", datetime(2022, 9, 15, 12, 0), "X", "Y", 1),
            Txn("self", datetime(2022, 9, 2), "X", "X", 1),
        ]
    )
    assert build_legs(conn, SPLIT) == 8  # 4 non-self txns x 2 legs
    got = dict(
        conn.execute("SELECT txn_id, window_start FROM legs WHERE direction = 'out'").fetchall()
    )
    assert got == {
        "a": date(2022, 9, 1), "b": date(2022, 9, 1), "c": date(2022, 9, 8), "d": date(2022, 9, 15),
    }  # fmt: skip


def test_window_straddling_split_is_rejected() -> None:
    # Rows are split at Sep 8, but windows anchored on Sep 10 run Sep 3-9 and would
    # contain both a train row (Sep 5) and a test row (Sep 9).
    conn = make_db(
        [Txn("a", datetime(2022, 9, 5), "X", "Y", 1), Txn("b", datetime(2022, 9, 9), "X", "Y", 1)]
    )
    with pytest.raises(ValueError, match="mix train and test"):
        build_legs(conn, date(2022, 9, 10))


# --- features ------------------------------------------------------------------


def test_features_for_hand_built_account() -> None:
    t = datetime(2022, 9, 2, 9, 0)
    conn = _features(
        [
            Txn("i1", t, "P", "H", 9500.0),  # near threshold
            Txn("i2", t + timedelta(hours=1), "Q", "H", 9000.0, cross_currency=True),  # near
            Txn("i3", t + timedelta(hours=2), "R", "H", 10000.0, fmt="Wire"),  # at threshold: no
            Txn("i4", t + timedelta(hours=3), "P", "H", 8999.0),  # just below band: no
            Txn("o1", t + timedelta(days=3), "H", "S", 2000.0),  # round
        ]
    )
    f = _row(conn, "H")
    assert (f["n_txn"], f["n_in"], f["n_out"]) == (5, 4, 1)
    assert (f["cp_in"], f["cp_out"]) == (3, 1)
    assert f["usd_in"] == pytest.approx(37499.0) and f["usd_out"] == pytest.approx(2000.0)
    assert f["n_near_threshold"] == 2
    assert f["cross_currency_share"] == pytest.approx(1 / 5)
    assert f["share_ach"] == pytest.approx(4 / 5) and f["share_wire"] == pytest.approx(1 / 5)
    assert f["round_share"] == pytest.approx(4 / 5)  # 9500, 9000, 10000, 2000
    assert f["max_24h"] == 4
    assert f["burst_ratio"] == pytest.approx(4 / (5 / 7))


def test_pass_through_window_amount_and_counterparty() -> None:
    t = datetime(2022, 9, 2, 0, 0)
    conn = _features(
        [
            Txn("in1", t, "P", "H", 1000.0),
            Txn("ok", t + timedelta(hours=47), "H", "S", 950.0),  # within 48h, -5%
            Txn("in2", t + timedelta(hours=1), "Q", "H", 5000.0),
            Txn("late", t + timedelta(hours=50), "H", "S", 5000.0),  # 49h after in2
            Txn("in3", t + timedelta(hours=2), "R", "H", 3000.0),
            Txn("back", t + timedelta(hours=3), "H", "R", 3000.0),  # returns to the sender
        ]
    )
    assert _row(conn, "H")["n_pass_through"] == 1
    ids = conn.execute(
        "SELECT pass_through_txn_ids FROM pass_through WHERE account = 'H'"
    ).fetchone()
    assert sorted(ids[0]) == ["in1", "ok"]


# --- rules ---------------------------------------------------------------------


def _hits(conn, th=ALL_ON):
    counts = build_rule_hits(conn, th)
    rows = conn.execute("SELECT account, rule_id, txn_ids FROM rule_hits").fetchall()
    return counts, {(a, r): ids for a, r, ids in rows}


def test_fan_in_and_structuring_return_triggering_txns() -> None:
    t = datetime(2022, 9, 3, 10, 0)
    txns = fan_in("f", "H", t, 3) + [
        Txn("s1", t, "Z", "K", 9500.0),
        Txn("s2", t + timedelta(days=2), "Z", "K", 9100.0),
        Txn("s3", t + timedelta(days=2), "Z", "K", 50.0),
    ]
    counts, hits = _hits(_features(txns))
    assert hits[("H", "FAN_IN")] == ["f0", "f1", "f2"]
    assert hits[("Z", "STRUCTURING")] == ["s1", "s2"]  # s3 not in the band
    assert ("K", "FAN_IN") not in hits  # only one distinct sender


def test_velocity_spike_cites_peak_24h_only() -> None:
    t = datetime(2022, 9, 2, 0, 0)
    burst = [Txn(f"v{i}", t + timedelta(hours=i), "V", f"C{i}", 10.0) for i in range(4)]
    quiet = [Txn("q", t + timedelta(days=4), "V", "C9", 10.0)]
    _, hits = _hits(_features(burst + quiet))
    assert hits[("V", "VELOCITY_SPIKE")] == ["v0", "v1", "v2", "v3"]


def test_disabled_rule_never_fires() -> None:
    conn = _features(fan_in("f", "H", datetime(2022, 9, 3), 5))
    counts, hits = _hits(conn, ALL_ON.model_copy(update={"fan_in_min_cp": None}))
    assert counts["FAN_IN"] == 0 and not any(r == "FAN_IN" for _, r in hits)


# --- anomaly -------------------------------------------------------------------


def _scored(txns):
    conn = _features(txns)
    fit_and_score(conn)
    return conn


def test_anomaly_model_is_fit_on_train_only() -> None:
    base = random_traffic()
    train_scores = lambda c: c.execute(  # noqa: E731
        "SELECT a.anomaly_score FROM anomaly_scores a JOIN account_window_features f "
        "USING (bank, account, window_start) WHERE f.split = 'train' ORDER BY ALL"
    ).fetchall()
    a = _scored(base)
    # A wild test-period account must not move the model or the train percentiles.
    b = _scored(base + fan_in("x", "TESTHUB", datetime(2022, 9, 10), 40))
    assert train_scores(a) == train_scores(b)
    pct = b.execute("SELECT anomaly_pct FROM anomaly_scores WHERE account = 'TESTHUB'").fetchone()[
        0
    ]
    assert pct == 1.0  # more anomalous than every train window


# --- records -------------------------------------------------------------------


def test_budget_threshold_respects_budget_and_ties() -> None:
    s = np.array([0.9, 0.8, 0.7, 0.7, 0.1] + [0.0] * 995)
    assert budget_threshold(s, 2) == 0.8  # top 2 of 1000
    assert budget_threshold(s, 3) == 0.8  # 0.7 tie would overflow 3 -> excluded
    assert budget_threshold(s, 4) == 0.7
    assert budget_threshold(np.zeros(1000), 10) == float("inf")


def test_alert_records_are_deterministic_pydantic_rows(tmp_path) -> None:
    def run():
        conn = _scored(random_traffic() + fan_in("f", "H", datetime(2022, 9, 3), 6))
        build_rule_hits(conn, ALL_ON)
        score_windows(conn, ALL_ON)
        build_alerts(conn, ALL_ON)
        return conn

    conn = run()
    alert_id = conn.execute(
        "SELECT alert_id FROM alerts WHERE account = 'H' AND window_start = DATE '2022-09-01'"
    ).fetchone()[0]
    alert = fetch_alert(conn, alert_id)
    assert alert.window_end == date(2022, 9, 7)
    assert "FAN_IN" in alert.triggered_rules
    assert set(alert.rule_txn_ids) >= {f"f{i}" for i in range(6)}
    expected = RULE_WEIGHT * min(
        1.0, 0.5 + (0.4 if "VELOCITY_SPIKE" in alert.triggered_rules else 0)
    )
    assert alert.combined_risk_score == pytest.approx(expected + ANOMALY_WEIGHT * alert.anomaly_pct)
    assert alert_id in {r[0] for r in run().execute("SELECT alert_id FROM alerts").fetchall()}

    path = tmp_path / "th.json"
    save_thresholds(ALL_ON, path)
    assert load_thresholds(path) == ALL_ON


def test_alerts_never_below_cutoff() -> None:
    conn = _scored(random_traffic())
    build_rule_hits(conn, ALL_ON)
    score_windows(conn, ALL_ON)
    build_alerts(conn, ALL_ON)
    assert (
        conn.execute(
            "SELECT count(*) FROM alerts WHERE combined_risk_score < ?", [ALL_ON.alert_threshold]
        ).fetchone()[0]
        == 0
    )
