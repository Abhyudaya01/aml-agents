import json
from datetime import date

from aml.data.schema import Customer
from aml.pii.tokenizer import PIIDetector, PIIMasker
from aml.pii.vault import PIIVault

ALICE = Customer(
    customer_id="CUST-0000001",
    bank="010",
    account="8000EBD30",
    name="Alice Example",
    address="12 Main St, Springfield, IL 62701",
    date_of_birth=date(1980, 5, 17),
    occupation="Teacher",
)


def test_mask_unmask_round_trip_nested() -> None:
    masker = PIIMasker(PIIVault())
    obj = {
        "customer": ALICE.model_dump(mode="json"),
        "txns": [{"from_account": "8000EBD30", "to_account": "800737690", "amount_usd": 10.5}],
        "note": "Alice Example sent funds from 8000EBD30 to 800AAF0B0.",
        "n_accounts": 2,
    }
    masked = masker.mask(obj)
    c = masked["customer"]
    assert c["name"].startswith("CUST_") and c["address"].startswith("ADDR_")
    assert c["date_of_birth"].startswith("DOB_") and c["account"].startswith("ACCT_")
    assert c["bank"] == "010" and c["occupation"] == "Teacher"  # not PII
    assert masked["txns"][0]["from_account"] == c["account"]  # same value, same token
    assert masked["n_accounts"] == 2  # ints under *account* keys are counts, not IDs
    assert "Alice" not in masked["note"] and "800AAF0B0" not in masked["note"]

    assert masker.unmask(json.dumps(masked)) == json.dumps(obj)


def test_mask_accepts_pydantic_models_and_unmask_restores_values() -> None:
    masker = PIIMasker(PIIVault())
    masked = masker.mask(ALICE)
    for field in ("name", "address", "account"):
        assert masker.unmask(masked[field]) == getattr(ALICE, field)
    assert masker.unmask(masked["date_of_birth"]) == "1980-05-17"


def test_tokens_are_stable_across_reopen(tmp_path) -> None:
    path = tmp_path / "vault.sqlite"
    first = PIIMasker(PIIVault(path)).mask(ALICE)
    second = PIIMasker(PIIVault(path)).mask(ALICE)
    assert first == second
    assert first["name"] == "CUST_0001"


def test_unknown_tokens_left_untouched() -> None:
    assert PIIMasker(PIIVault()).unmask("see CUST_9999") == "see CUST_9999"


def test_detector_flags_raw_values_and_account_ids() -> None:
    detector = PIIDetector.from_customers([ALICE.model_dump()])
    assert detector.find("customer CUST_0001 at ADDR_0001") == []
    assert detector.find("customer Alice Example") == ["Alice Example"]
    assert detector.find("acct 800FFFFF1") == ["800FFFFF1"]  # any account-shaped ID
    assert detector.find("born 1980-05-17") == ["1980-05-17"]
