"""Reversible PII vault (SQLite).

Each (kind, value) pair gets one stable token, e.g. CUST_0042. Tokens are numbered
per kind in first-seen order and never reused, so the same value maps to the same
token across runs as long as the vault file is kept.
"""

import sqlite3
from pathlib import Path

PREFIXES: dict[str, str] = {
    "name": "CUST",
    "address": "ADDR",
    "dob": "DOB",
    "account": "ACCT",
}
TOKEN_PATTERN = r"\b(?:CUST|ADDR|DOB|ACCT)_\d{4,}\b"


class PIIVault:
    def __init__(self, path: Path | str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS vault (
                token TEXT PRIMARY KEY,
                kind  TEXT NOT NULL,
                seq   INTEGER NOT NULL,
                value TEXT NOT NULL,
                UNIQUE (kind, value),
                UNIQUE (kind, seq)
            )
            """
        )
        self._db.commit()

    def tokenize(self, kind: str, value: str) -> str:
        prefix = PREFIXES[kind]
        row = self._db.execute(
            "SELECT token FROM vault WHERE kind = ? AND value = ?", [kind, value]
        ).fetchone()
        if row:
            return row[0]
        with self._db:
            (seq,) = self._db.execute(
                "SELECT coalesce(max(seq), 0) + 1 FROM vault WHERE kind = ?", [kind]
            ).fetchone()
            token = f"{prefix}_{seq:04d}"
            self._db.execute("INSERT INTO vault VALUES (?, ?, ?, ?)", [token, kind, seq, value])
        return token

    def detokenize(self, token: str) -> str | None:
        row = self._db.execute("SELECT value FROM vault WHERE token = ?", [token]).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._db.close()
