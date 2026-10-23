"""Gateway: every call masked, checked for raw PII, then audited."""

import json
from datetime import date

import pytest
from pydantic import BaseModel

from aml.audit.log import AuditLog
from aml.data.customers import generate_customers
from aml.llm.gateway import LLMGateway, PromptSpec
from aml.llm.pricing import ModelPrice
from aml.llm.providers.fake import FakeLLM
from aml.llm.types import StructuredOutputError
from aml.pii.tokenizer import PIIDetector, PIILeakError, PIIMasker
from aml.pii.vault import PIIVault

PRICE = ModelPrice(input_per_mtok=4.0, output_per_mtok=20.0)
ACCOUNTS = [("010", "8000EBD30"), ("0220", "800737690"), ("03208", "8000F4580")]


class Summary(BaseModel):
    subject: str
    text: str


@pytest.fixture
def customers():
    # Seeded fake profiles, built exactly as ingest builds the customers table.
    return [c.model_dump() for c in generate_customers(ACCOUNTS, seed=42)]


def _gateway(llm, customers, audit=None):
    audit = audit or AuditLog()
    detector = PIIDetector.from_customers(customers)
    return LLMGateway(llm, PIIMasker(PIIVault()), audit, PRICE, 2, detector), audit


def _raw_values(customers) -> list[str]:
    out = []
    for c in customers:
        dob = c["date_of_birth"]
        out += [c["name"], c["address"], c["account"],
                dob.isoformat() if isinstance(dob, date) else str(dob)]  # fmt: skip
    return out


def _spec(customers) -> PromptSpec:
    subject = customers[0]
    return PromptSpec(
        system="You are an AML analyst.",
        instructions="Summarize the subject's activity.",
        context={
            "subject": subject,
            "counterparties": customers[1:],
            "analyst_note": f"{subject['name']} of {subject['address']} sent funds to "
            f"{customers[1]['account']}.",
        },
    )


def test_no_raw_pii_in_logged_or_sent_prompt(customers) -> None:
    def echo_subject(prompt, schema):
        token = json.loads(prompt.user.split("<context>")[1].split("</context>")[0])
        return Summary(subject=token["subject"]["name"], text="Fan-in pattern.")

    llm = FakeLLM([echo_subject])
    gw, audit = _gateway(llm, customers)
    result = gw.generate("CASE-1", "narrative", _spec(customers), Summary)

    (event,) = audit.events()
    logged = event.payload_json
    sent = llm.prompts[0].system + llm.prompts[0].user
    for raw in _raw_values(customers):
        assert raw not in logged, raw
        assert raw not in sent, raw
    assert "CUST_" in sent and "ACCT_" in sent
    assert result.output.subject.startswith("CUST_")  # output stays tokenized
    assert audit.verify_chain().ok


def test_audit_event_records_model_usage_cost_latency(customers) -> None:
    llm = FakeLLM([Summary(subject="CUST_0001", text="ok")])
    gw, audit = _gateway(llm, customers)
    result = gw.generate("CASE-1", "narrative", _spec(customers), Summary)
    (e,) = audit.events()
    assert (e.event_type, e.actor, e.case_id, e.model_id) == (
        "llm_call",
        "narrative",
        "CASE-1",
        "fake",
    )
    assert json.loads(e.tokens) == {"input": result.usage.input_tokens,
                                    "output": result.usage.output_tokens}  # fmt: skip
    assert e.cost == pytest.approx(result.cost_usd) and e.latency == pytest.approx(
        result.latency_ms
    )
    assert len(e.prompt_hash) == 64 and e.payload["schema"] == "Summary"


def test_raw_pii_in_prompt_is_blocked_and_audited(customers) -> None:
    llm = FakeLLM([Summary(subject="x", text="y")])
    gw, audit = _gateway(llm, customers)
    # A developer mistake: raw PII typed into instructions under a non-PII key name.
    leaky = PromptSpec(
        system="s", instructions="Look at this:", context={"blob": customers[2]["address"]}
    )
    with pytest.raises(PIILeakError):
        gw.generate("CASE-2", "triage", leaky, Summary)
    assert llm.prompts == []  # never reached the model
    (e,) = audit.events()
    assert e.event_type == "llm_call_blocked" and customers[2]["address"] not in e.payload_json


def test_validation_failure_is_audited(customers) -> None:
    llm = FakeLLM(["{}"] * 3)
    gw, audit = _gateway(llm, customers)
    with pytest.raises(StructuredOutputError):
        gw.generate("CASE-3", "triage", _spec(customers), Summary)
    (e,) = audit.events()
    assert e.event_type == "llm_call_failed" and e.payload["attempts"] == 3


def test_every_call_appends_one_event(customers) -> None:
    llm = FakeLLM([Summary(subject="a", text="b")] * 3)
    gw, audit = _gateway(llm, customers)
    for i in range(3):
        gw.generate(f"CASE-{i}", "triage", _spec(customers), Summary)
    assert [e.case_id for e in audit.events()] == ["CASE-0", "CASE-1", "CASE-2"]
