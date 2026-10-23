"""Rule-threshold tuning on the TRAIN split only.

Objective: maximize train recall (positive account-windows alerted) subject to an
alert budget of N alerts per 1,000 train account-windows. Ties go to higher
precision.

Search: coordinate ascent. Each parameter's candidates are quantiles (p90 ... p99.9)
of that feature over train windows where it is non-zero, plus "disabled". For
every candidate configuration the combined-score cutoff is re-derived from the
budget (label-free), then recall is measured against train labels.

The output is a RuleThresholds JSON; that file is the only thing detection code
ever sees from this module.
"""

import math
from datetime import date

import duckdb
from pydantic import BaseModel

from aml.alerts.records import budget_threshold, combined_score_sql
from aml.alerts.thresholds import RuleThresholds
from aml.eval.alert_metrics import POSITIVE_UNITS, build_positive_units

# threshold field -> (feature column, integer-valued?)
PARAMS: dict[str, tuple[str, bool]] = {
    "fan_in_min_cp": ("cp_in", True),
    "fan_out_min_cp": ("cp_out", True),
    "structuring_min_count": ("n_near_threshold", True),
    "velocity_min_24h": ("max_24h", True),
    "velocity_min_burst": ("burst_ratio", False),
    "pass_through_min_count": ("n_pass_through", True),
}
QUANTILES = (0.90, 0.95, 0.99, 0.995, 0.999)
START_QUANTILE = 0.99
MAX_PASSES = 4


class TrialResult(BaseModel):
    params: dict[str, float | None]
    alert_threshold: float
    alerts: int
    recall: float
    precision: float


class TuningResult(BaseModel):
    thresholds: RuleThresholds
    train_windows: int
    train_positive_windows: int
    best: TrialResult
    trials: int


def candidate_values(conn: duckdb.DuckDBPyConnection, column: str, integer: bool) -> list:
    qs = conn.execute(
        f"SELECT quantile_cont({column}, {list(QUANTILES)}) FROM account_window_features "
        f"WHERE split = 'train' AND {column} > 0"
    ).fetchone()[0]
    if qs is None:
        return [None]
    values = [float(math.ceil(q)) if integer else round(float(q), 2) for q in qs]
    return sorted(set(values)) + [None]


def _thresholds(params: dict, tau: float, budget: float) -> RuleThresholds:
    return RuleThresholds(**params, alert_threshold=tau, budget_per_1k_windows=budget)


def _evaluate(conn: duckdb.DuckDBPyConnection, params: dict, budget: float) -> TrialResult:
    data = conn.execute(
        f"""
        SELECT {combined_score_sql(_thresholds(params, 0.0, budget))} AS score,
               p.bank IS NOT NULL AS positive
        FROM account_window_features f
        JOIN anomaly_scores a USING (bank, account, window_start)
        LEFT JOIN {POSITIVE_UNITS} p USING (bank, account, window_start)
        WHERE f.split = 'train'
        """
    ).fetchnumpy()
    score, y = data["score"], data["positive"]
    tau = budget_threshold(score, budget)
    alerted = score >= tau
    tp = int((alerted & y).sum())
    n_alerts = int(alerted.sum())
    return TrialResult(
        params=params,
        alert_threshold=tau,
        alerts=n_alerts,
        recall=tp / max(int(y.sum()), 1),
        precision=tp / n_alerts if n_alerts else 0.0,
    )


def tune_thresholds(
    conn: duckdb.DuckDBPyConnection, split_date: date, budget_per_1k: float
) -> TuningResult:
    build_positive_units(conn, split_date)
    candidates = {p: candidate_values(conn, col, is_int) for p, (col, is_int) in PARAMS.items()}
    start_idx = QUANTILES.index(START_QUANTILE)
    current = {p: c[min(start_idx, len(c) - 1)] for p, c in candidates.items()}
    best = _evaluate(conn, current, budget_per_1k)
    trials = 1
    for _ in range(MAX_PASSES):
        improved = False
        for param, values in candidates.items():
            for value in values:
                if value == current[param]:
                    continue
                trial = _evaluate(conn, {**current, param: value}, budget_per_1k)
                trials += 1
                if (trial.recall, trial.precision) > (best.recall, best.precision):
                    best, current, improved = trial, trial.params, True
        if not improved:
            break

    n_train, n_pos = conn.execute(
        f"""
        SELECT count(*), count(p.bank) FROM account_window_features f
        LEFT JOIN {POSITIVE_UNITS} p USING (bank, account, window_start)
        WHERE f.split = 'train'
        """
    ).fetchone()
    return TuningResult(
        thresholds=_thresholds(best.params, best.alert_threshold, budget_per_1k),
        train_windows=n_train,
        train_positive_windows=n_pos,
        best=best,
        trials=trials,
    )
