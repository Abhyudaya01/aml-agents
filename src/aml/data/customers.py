"""Synthetic customer profiles, one per (bank, account).

HI-Small has no customer data. Later phases need names, addresses, etc. to
exercise the PII tokenizer, so we generate fake ones with a seeded Faker.

Accounts are keyed by (bank, account): 8 account IDs in HI-Small appear under two
different banks. Profiles are generated in sorted key order with a fixed seed, so
the same input always yields the same profiles (for a given Faker version, which
uv.lock pins).
"""

import csv
import tempfile
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import duckdb
from faker import Faker

from aml.data.schema import Customer

DOB_RANGE = (date(1935, 1, 1), date(2004, 12, 31))  # fixed, so output never depends on today


def generate_customers(accounts: Iterable[tuple[str, str]], seed: int) -> list[Customer]:
    fake = Faker("en_US")
    fake.seed_instance(seed)
    return [
        Customer(
            customer_id=f"CUST-{i:07d}",
            bank=bank,
            account=account,
            name=fake.name(),
            address=fake.address().replace("\n", ", "),
            date_of_birth=fake.date_between_dates(*DOB_RANGE),
            occupation=fake.job(),
        )
        for i, (bank, account) in enumerate(sorted(set(accounts)))
    ]


def distinct_accounts(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    return conn.execute(
        """
        SELECT from_bank, from_account FROM transactions
        UNION SELECT to_bank, to_account FROM transactions
        ORDER BY 1, 2
        """
    ).fetchall()


def build_customers(conn: duckdb.DuckDBPyConnection, seed: int) -> int:
    customers = generate_customers(distinct_accounts(conn), seed)
    fields = list(Customer.model_fields)
    # Bulk-load via a temp CSV: far faster than row-by-row inserts for ~500k rows.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "customers.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows([getattr(c, k) for k in fields] for c in customers)
        conn.execute(
            """
            CREATE OR REPLACE TABLE customers (
                customer_id VARCHAR PRIMARY KEY, bank VARCHAR, account VARCHAR,
                name VARCHAR, address VARCHAR, date_of_birth DATE, occupation VARCHAR,
                UNIQUE (bank, account)
            )
            """
        )
        conn.execute(
            "INSERT INTO customers SELECT * FROM read_csv(?, header = true, all_varchar = true)",
            [str(path)],
        )
    return len(customers)
