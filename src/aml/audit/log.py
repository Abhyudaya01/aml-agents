"""Append-only, hash-chained audit log (SQLite).

Every event stores prev_hash (the previous event's hash) and
hash = sha256(canonical JSON of all its other fields). Editing, deleting, inserting
or reordering any row breaks the chain, and verify_chain() reports the first bad
event. SQLite triggers additionally reject UPDATE and DELETE, so tampering requires
raw file access (dropping the triggers), which the chain then exposes.

Limitation: truncating the newest events leaves a valid shorter chain. Anchoring the
latest hash outside the database (e.g. in the case record) would close that gap.
"""

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

GENESIS_HASH = "0" * 64
FIELDS = (
    "seq", "event_id", "ts", "case_id", "actor", "event_type", "payload_json",
    "model_id", "prompt_hash", "tokens", "cost", "latency", "prev_hash",
)  # fmt: skip


class AuditEvent(BaseModel):
    seq: int
    event_id: str
    ts: str
    case_id: str
    actor: str
    event_type: str
    payload_json: str
    model_id: str | None = None
    prompt_hash: str | None = None
    tokens: str | None = None  # JSON: {"input": n, "output": m}
    cost: float | None = None
    latency: float | None = None  # milliseconds
    prev_hash: str
    hash: str

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)


class ChainReport(BaseModel):
    ok: bool
    n_events: int
    first_bad_seq: int | None = None
    reason: str | None = None


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def event_hash(fields: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json({k: fields[k] for k in FIELDS}).encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class AuditLog:
    def __init__(self, path: Path | str = ":memory:", clock: Callable[[], str] = _utc_now):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), isolation_level=None)
        self._clock = clock
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL, ts TEXT NOT NULL,
                case_id TEXT NOT NULL, actor TEXT NOT NULL, event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL, model_id TEXT, prompt_hash TEXT, tokens TEXT,
                cost REAL, latency REAL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
            BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
            BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
            """
        )

    def append(
        self,
        case_id: str,
        actor: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        model_id: str | None = None,
        prompt_hash: str | None = None,
        tokens: dict[str, int] | None = None,
        cost: float | None = None,
        latency: float | None = None,
    ) -> AuditEvent:
        self._db.execute("BEGIN IMMEDIATE")  # serialize: read head + insert atomically
        try:
            last = self._db.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1")
            row = last.fetchone()
            fields: dict[str, Any] = {
                "seq": (row[0] + 1) if row else 1,
                "event_id": uuid.uuid4().hex,
                "ts": self._clock(),
                "case_id": case_id,
                "actor": actor,
                "event_type": event_type,
                "payload_json": canonical_json(payload),
                "model_id": model_id,
                "prompt_hash": prompt_hash,
                "tokens": canonical_json(tokens) if tokens is not None else None,
                "cost": cost,
                "latency": latency,
                "prev_hash": row[1] if row else GENESIS_HASH,
            }
            fields["hash"] = event_hash(fields)
            cols = (*FIELDS, "hash")
            self._db.execute(
                f"INSERT INTO events ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [fields[c] for c in cols],
            )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return AuditEvent(**fields)

    def events(self, case_id: str | None = None) -> list[AuditEvent]:
        cols = (*FIELDS, "hash")
        sql = f"SELECT {', '.join(cols)} FROM events"
        args: list[Any] = []
        if case_id is not None:
            sql += " WHERE case_id = ?"
            args.append(case_id)
        rows = self._db.execute(sql + " ORDER BY seq", args).fetchall()
        return [AuditEvent(**dict(zip(cols, r, strict=True))) for r in rows]

    def verify_chain(self) -> ChainReport:
        events = self.events()
        prev = GENESIS_HASH
        for expected_seq, e in enumerate(events, start=1):
            fields = e.model_dump()
            if e.seq != expected_seq:
                return ChainReport(
                    ok=False, n_events=len(events), first_bad_seq=e.seq,
                    reason=f"sequence gap: expected {expected_seq}",
                )  # fmt: skip
            if e.prev_hash != prev:
                return ChainReport(
                    ok=False, n_events=len(events), first_bad_seq=e.seq,
                    reason="prev_hash does not match previous event",
                )  # fmt: skip
            if event_hash(fields) != e.hash:
                return ChainReport(
                    ok=False, n_events=len(events), first_bad_seq=e.seq,
                    reason="hash does not match event contents",
                )  # fmt: skip
            prev = e.hash
        return ChainReport(ok=True, n_events=len(events))

    def close(self) -> None:
        self._db.close()


def record_tool_call(
    log: AuditLog, case_id: str, actor: str, tool: str, args: dict, result: Any
) -> AuditEvent:
    return log.append(case_id, actor, "tool_call", {"tool": tool, "args": args, "result": result})


def record_human_decision(
    log: AuditLog, case_id: str, reviewer: str, decision: str, note: str = ""
) -> AuditEvent:
    return log.append(
        case_id, f"human:{reviewer}", "human_decision", {"decision": decision, "note": note}
    )
