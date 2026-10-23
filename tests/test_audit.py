import itertools
import sqlite3

import pytest

from aml.audit.log import AuditLog, record_human_decision, record_tool_call


def _clock():
    counter = itertools.count()
    return lambda: f"2026-10-23T12:00:00.{next(counter):06d}+00:00"


@pytest.fixture
def log_path(tmp_path):
    path = tmp_path / "audit.sqlite"
    log = AuditLog(path, clock=_clock())
    tokens = {"input": 3, "output": 2}
    log.append(
        "C1", "triage", "llm_call", {"x": 1}, model_id="fake", tokens=tokens, cost=0.0, latency=1.5
    )
    record_tool_call(
        log, "C1", "investigation", "get_counterparties", {"acct": "ACCT_0001"}, [1, 2]
    )
    record_human_decision(log, "C1", "analyst1", "escalate", "looks like a fan-in")
    log.close()
    return path


def _raw(path):
    """Simulate someone with direct file access: drop the append-only triggers."""
    db = sqlite3.connect(str(path))
    db.execute("DROP TRIGGER events_no_update")
    db.execute("DROP TRIGGER events_no_delete")
    return db


def test_chain_verifies(log_path) -> None:
    report = AuditLog(log_path).verify_chain()
    assert report.ok and report.n_events == 3


def test_events_link_to_previous_hash(log_path) -> None:
    events = AuditLog(log_path).events()
    assert events[0].prev_hash == "0" * 64
    assert [e.prev_hash for e in events[1:]] == [e.hash for e in events[:-1]]
    assert events[2].actor == "human:analyst1" and events[2].payload["decision"] == "escalate"


def test_update_and_delete_are_rejected(log_path) -> None:
    db = sqlite3.connect(str(log_path))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.execute("UPDATE events SET actor = 'x' WHERE seq = 1")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.execute("DELETE FROM events WHERE seq = 1")


def test_edited_row_breaks_chain(log_path) -> None:
    db = _raw(log_path)
    db.execute(
        """UPDATE events SET payload_json = '{"decision":"close","note":""}' WHERE seq = 3"""
    )
    db.commit()
    report = AuditLog(log_path).verify_chain()
    assert not report.ok and report.first_bad_seq == 3 and "contents" in report.reason


def test_edited_row_with_recomputed_hash_breaks_next_link(log_path) -> None:
    from aml.audit.log import event_hash

    log = AuditLog(log_path)
    e = log.events()[0]
    forged = {**e.model_dump(), "cost": 99.0}
    db = _raw(log_path)
    db.execute("UPDATE events SET cost = 99.0, hash = ? WHERE seq = 1", [event_hash(forged)])
    db.commit()
    report = AuditLog(log_path).verify_chain()
    assert not report.ok and report.first_bad_seq == 2 and "prev_hash" in report.reason


def test_deleted_row_breaks_chain(log_path) -> None:
    db = _raw(log_path)
    db.execute("DELETE FROM events WHERE seq = 2")
    db.commit()
    report = AuditLog(log_path).verify_chain()
    assert not report.ok and report.first_bad_seq == 3


def test_reordered_rows_break_chain(log_path) -> None:
    db = _raw(log_path)
    db.execute("UPDATE events SET seq = 99 WHERE seq = 1")
    db.execute("UPDATE events SET seq = 1 WHERE seq = 2")
    db.execute("UPDATE events SET seq = 2 WHERE seq = 99")
    db.commit()
    assert not AuditLog(log_path).verify_chain().ok


def test_append_after_reopen_continues_chain(log_path) -> None:
    log = AuditLog(log_path)
    e = log.append("C2", "triage", "llm_call", {})
    assert e.seq == 4 and log.verify_chain().ok
