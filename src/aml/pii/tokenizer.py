"""mask(obj) / unmask(text), plus a detector for raw PII that slipped through.

Masking is structural first: values under PII-bearing keys (name, address,
date_of_birth, *account*) are replaced by vault tokens. A free-text pass then
replaces, inside every string, (a) any PII value this masker has already tokenized
and (b) anything shaped like an HI-Small account ID.

Bank codes are institution identifiers, not personal data, and stay as-is.
"""

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from pydantic import BaseModel

from aml.pii.vault import TOKEN_PATTERN, PIIVault

# HI-Small account IDs: 9 upper-case hex chars beginning with 8 (e.g. 8000EBD30).
ACCOUNT_PATTERN = re.compile(r"\b8[0-9A-F]{8}\b")
_TOKEN_RE = re.compile(TOKEN_PATTERN)


def pii_kind(key: str) -> str | None:
    k = key.lower()
    if k in ("name", "customer_name", "full_name"):
        return "name"
    if k == "address":
        return "address"
    if k in ("date_of_birth", "dob"):
        return "dob"
    if "account" in k:
        return "account"
    return None


class PIIMasker:
    def __init__(self, vault: PIIVault) -> None:
        self._vault = vault
        self._seen: dict[str, str] = {}  # raw value -> token, for the free-text pass
        self._seen_re: re.Pattern[str] | None = None

    # -- public API ---------------------------------------------------------------

    def mask(self, obj: Any) -> Any:
        """Return a JSON-compatible copy of obj with PII replaced by tokens."""
        if isinstance(obj, BaseModel):
            obj = obj.model_dump(mode="json")
        structural = self._mask_structure(obj)
        return self._mask_strings(structural)

    def mask_text(self, text: str) -> str:
        return self._mask_strings(text)

    def unmask(self, text: str) -> str:
        """Replace tokens with their original values. Final human-facing render only."""
        return _TOKEN_RE.sub(lambda m: self._vault.detokenize(m.group(0)) or m.group(0), text)

    # -- internals ----------------------------------------------------------------

    def _token(self, kind: str, value: Any) -> str:
        raw = value.isoformat() if isinstance(value, date) else str(value)
        token = self._vault.tokenize(kind, raw)
        if raw not in self._seen:
            self._seen[raw] = token
            self._seen_re = None
        return token

    def _mask_structure(self, obj: Any, kind: str | None = None) -> Any:
        # A key's kind applies to its value and to list items under it, but not to the
        # keys of a nested dict (e.g. {"accounts": [{"bank": ...}]} keeps bank as-is).
        if isinstance(obj, dict):
            return {k: self._mask_structure(v, pii_kind(str(k))) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [self._mask_structure(v, kind) for v in obj]
        if kind is not None and obj is not None and not isinstance(obj, bool | int | float):
            return self._token(kind, obj)
        if isinstance(obj, date):
            return obj.isoformat()
        return obj

    def _mask_strings(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._mask_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._mask_strings(v) for v in obj]
        if not isinstance(obj, str):
            return obj
        text = obj
        if self._seen:
            if self._seen_re is None:
                alts = sorted(self._seen, key=len, reverse=True)
                self._seen_re = re.compile("|".join(re.escape(a) for a in alts))
            text = self._seen_re.sub(lambda m: self._seen[m.group(0)], text)
        return ACCOUNT_PATTERN.sub(lambda m: self._token("account", m.group(0)), text)


class PIILeakError(RuntimeError):
    """Raw PII was found in text about to leave the process."""


class PIIDetector:
    """Exact-match detector over a known set of raw PII values (e.g. the customers
    table). Used as a last check on rendered prompts; it is a safety net, not the
    masking mechanism.
    """

    def __init__(self, values: Iterable[str]) -> None:
        vals = sorted({v for v in values if v and len(v) >= 4}, key=len, reverse=True)
        self._re = re.compile("|".join(re.escape(v) for v in vals)) if vals else None

    @classmethod
    def from_customers(cls, rows: Iterable[dict[str, Any]]) -> "PIIDetector":
        values: list[str] = []
        for r in rows:
            values += [str(r["name"]), str(r["address"]), str(r["account"])]
            dob = r["date_of_birth"]
            values.append(dob.isoformat() if isinstance(dob, date) else str(dob))
        return cls(values)

    def find(self, text: str) -> list[str]:
        hits = set(ACCOUNT_PATTERN.findall(text))
        if self._re is not None:
            hits |= set(self._re.findall(text))
        return sorted(hits)
