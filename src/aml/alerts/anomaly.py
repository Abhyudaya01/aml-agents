"""IsolationForest anomaly score per account-window.

Fit on train-split windows only. Scores are converted to percentiles against the
train score distribution (also train-only), so a test window's percentile answers
"how unusual is this compared with the training week?".
"""

from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from aml.alerts.features import MODEL_FEATURES

N_ESTIMATORS = 200
RANDOM_STATE = 0


def _matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    # All features are non-negative counts, amounts or shares; log1p tames the
    # heavy tails (hub accounts reach 100k+ legs) so splits aren't wasted on them.
    return np.log1p(np.column_stack([data[c].astype(float) for c in MODEL_FEATURES]))


def fit_and_score(conn: duckdb.DuckDBPyConnection, model_path: Path | None = None) -> int:
    """Create anomaly_scores(bank, account, window_start, anomaly_score, anomaly_pct)."""
    cols = ", ".join(MODEL_FEATURES)
    data = conn.execute(
        f"SELECT bank, account, window_start, split, {cols} FROM account_window_features "
        "ORDER BY bank, account, window_start"
    ).fetchnumpy()
    X = _matrix(data)
    is_train = data["split"] == "train"
    if not is_train.any():
        raise ValueError("no train windows to fit the anomaly model on")

    model = IsolationForest(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X[is_train])
    score = -model.score_samples(X)  # higher = more anomalous
    train_sorted = np.sort(score[is_train])
    pct = np.searchsorted(train_sorted, score, side="right") / len(train_sorted)

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    conn.register(
        "_scores",
        {
            "bank": data["bank"],
            "account": data["account"],
            "window_start": data["window_start"],
            "anomaly_score": score,
            "anomaly_pct": pct,
        },
    )
    conn.execute("CREATE OR REPLACE TABLE anomaly_scores AS SELECT * FROM _scores")
    conn.unregister("_scores")
    return len(score)
