"""Invariant 1: only src/aml/eval/ may touch labels.

Scans every Python file under src/aml except eval/ for label identifiers, the
eval DuckDB schema, and imports of aml.eval.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "aml"
ALLOWED_DIR = SRC / "eval"

FORBIDDEN = re.compile(
    r"is_laundering"
    r"|laundering_attempts"
    r"|Is Laundering"
    r"|txn_labels"
    r"|\baml\.eval\b"  # importing the eval package
    r"|\beval\.\w+"  # querying the eval schema (eval.txn_labels, ...)
)


def find_leaks(root: Path, allowed: Path) -> list[str]:
    leaks = []
    for path in sorted(root.rglob("*.py")):
        if allowed in path.parents:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if m := FORBIDDEN.search(line):
                leaks.append(f"{path.relative_to(root)}:{lineno}: {m.group(0)}")
    return leaks


def test_no_label_access_outside_eval() -> None:
    assert find_leaks(SRC, ALLOWED_DIR) == []


def test_scanner_catches_planted_leak(tmp_path: Path) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "ok.py").write_text("SELECT is_laundering FROM eval.txn_labels\n")
    (tmp_path / "alerts").mkdir()
    (tmp_path / "alerts" / "bad.py").write_text("from aml.eval.labels import x\n")
    assert find_leaks(tmp_path, tmp_path / "eval") == ["alerts/bad.py:1: aml.eval"]
