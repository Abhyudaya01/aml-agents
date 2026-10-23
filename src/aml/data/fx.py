"""Static currency -> USD rate table.

Source of the rates: the dataset itself. HI-Small contains 72,170 cross-currency
transactions where `amount_paid` (payment currency) and `amount_received`
(receiving currency) differ. For each currency C we took the median of
amount_received / amount_paid over rows paid in C and received in US Dollar.
Dispersion is tight (coefficient of variation < 0.1% for EUR/GBP/SAR/BTC, < 10% for
all), so a single static rate per currency is a faithful approximation.

We deliberately do NOT use real-world rates: they would contradict the simulator's
own cross-currency rows. `fx_consistency` re-checks the table against the data at
every ingest.
"""

import duckdb

USD_PER_UNIT: dict[str, float] = {
    "US Dollar": 1.0,
    "Euro": 1.17178,
    "UK Pound": 1.29166,
    "Swiss Franc": 1.09290,
    "Canadian Dollar": 0.757978,
    "Australian Dollar": 0.707814,
    "Yuan": 0.149307,
    "Yen": 0.00948767,
    "Rupee": 0.0136158,
    "Ruble": 0.0128528,
    "Shekel": 0.296121,
    "Saudi Riyal": 0.266588,
    "Mexican Peso": 0.0472968,
    "Brazil Real": 0.177101,
    "Bitcoin": 11881.6,
}


def load_fx_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE OR REPLACE TABLE fx_rates (currency VARCHAR PRIMARY KEY, usd_per_unit DOUBLE)"
    )
    conn.executemany("INSERT INTO fx_rates VALUES (?, ?)", list(USD_PER_UNIT.items()))


def unknown_currencies(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Currencies present in `transactions` with no rate in the table."""
    rows = conn.execute(
        """
        SELECT DISTINCT c FROM (
            SELECT receiving_currency AS c FROM transactions
            UNION SELECT payment_currency FROM transactions
        ) WHERE c NOT IN (SELECT currency FROM fx_rates) ORDER BY c
        """
    ).fetchall()
    return [r[0] for r in rows]


def fx_consistency(conn: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """For cross-currency rows, |received_usd / paid_usd - 1|: median and p95.

    Near zero means the static table agrees with the simulator's implied rates.
    """
    median, p95, n = conn.execute(
        """
        SELECT median(err), quantile_cont(err, 0.95), count(*) FROM (
            SELECT abs(amount_received_usd / amount_paid_usd - 1) AS err
            FROM transactions
            WHERE receiving_currency <> payment_currency AND amount_paid_usd > 0
        )
        """
    ).fetchone()
    return {"rows": float(n), "median_rel_err": float(median or 0), "p95_rel_err": float(p95 or 0)}
