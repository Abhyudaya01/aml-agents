"""Provider layer: retries, usage/cost accounting, and response parsing (no network)."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from aml.config import Settings
from aml.llm.factory import make_gateway, price_table
from aml.llm.pricing import ModelPrice, UnpricedModelError, cost_usd
from aml.llm.providers.anthropic import AnthropicProvider
from aml.llm.providers.base import generate_structured
from aml.llm.providers.fake import FakeLLM
from aml.llm.types import LLMRefusalError, Prompt, StructuredOutputError, Usage


class Triage(BaseModel):
    priority: int = Field(ge=1, le=5)
    reason: str


PROMPT = Prompt(system="sys", user="classify")
PRICE = ModelPrice(input_per_mtok=4.0, output_per_mtok=20.0)


def test_valid_first_try() -> None:
    llm = FakeLLM([Triage(priority=2, reason="fan-in")])
    r = generate_structured(llm, PROMPT, Triage, PRICE, max_retries=2)
    assert r.output == Triage(priority=2, reason="fan-in") and r.attempts == 1
    assert r.model_id == "fake" and r.latency_ms >= 0


def test_retries_on_validation_failure_with_feedback() -> None:
    llm = FakeLLM(['{"priority": 9, "reason": "x"}', '{"priority": 3, "reason": "ok"}'])
    r = generate_structured(llm, PROMPT, Triage, PRICE, max_retries=2)
    assert r.output.priority == 3 and r.attempts == 2
    retry_prompt = llm.prompts[1].user
    assert '"priority": 9' in retry_prompt and "less than or equal to 5" in retry_prompt


def test_gives_up_after_retry_budget_and_keeps_usage() -> None:
    llm = FakeLLM(["not json"] * 3)
    with pytest.raises(StructuredOutputError) as exc:
        generate_structured(llm, PROMPT, Triage, PRICE, max_retries=2)
    assert exc.value.attempts == 3 and exc.value.usage.output_tokens == 3 * 2


def test_usage_and_cost_accumulate_across_attempts() -> None:
    llm = FakeLLM(['{"priority": 0, "reason": "x"}', Triage(priority=1, reason="y")])
    r = generate_structured(llm, PROMPT, Triage, PRICE, max_retries=1)
    assert len(llm.prompts) == 2
    assert r.usage.input_tokens == sum(-(-len(p.system + p.user) // 4) for p in llm.prompts)
    assert r.cost_usd == pytest.approx(cost_usd(r.usage, PRICE))


def test_cost_formula() -> None:
    assert cost_usd(Usage(input_tokens=1_000_000, output_tokens=500_000), PRICE) == 14.0


def test_unpriced_model_rejected_at_build_time() -> None:
    from aml.audit.log import AuditLog
    from aml.pii.tokenizer import PIIMasker
    from aml.pii.vault import PIIVault

    settings = Settings(_env_file=None, llm_provider="openai", model_id="some-unpriced-model")
    with pytest.raises(UnpricedModelError, match="AML_LLM_PRICE_OVERRIDES"):
        make_gateway(
            settings, PIIMasker(PIIVault()), AuditLog(),
            provider=FakeLLM([], model_id="some-unpriced-model"),
        )  # fmt: skip


def test_price_overrides_extend_builtin_table() -> None:
    s = Settings(
        _env_file=None,
        llm_price_overrides={"my-model": {"input_per_mtok": 1, "output_per_mtok": 2}},
    )
    table = price_table(s)
    assert table["my-model"].output_per_mtok == 2 and table["claude-opus-5-5"].input_per_mtok == 4


# --- Anthropic provider response handling, against a stub client ----------------


def _stub_anthropic(body: dict):
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(http_response=SimpleNamespace(json=lambda: body))

    client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(with_raw_response=SimpleNamespace(parse=parse))
        )
    )
    return client, calls


def test_anthropic_provider_request_and_usage() -> None:
    body = {
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": '{"a": 1}'}],
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 7},
    }
    client, calls = _stub_anthropic(body)
    c = AnthropicProvider("claude-opus-5-5", "medium", 16000, client=client).complete(
        PROMPT, Triage
    )
    assert c.text == '{"a": 1}' and c.usage == Usage(input_tokens=15, output_tokens=7)
    sent = calls[0]
    assert sent["output_format"] is Triage and sent["output_config"] == {"effort": "medium"}
    assert sent["fallbacks"] == "default" and "temperature" not in sent


def test_anthropic_refusal_raises() -> None:
    client, _ = _stub_anthropic(
        {"stop_reason": "refusal", "stop_details": {"category": "cyber"}, "content": []}
    )
    with pytest.raises(LLMRefusalError, match="cyber"):
        AnthropicProvider("claude-opus-5-5", "medium", 16000, client=client).complete(
            PROMPT, Triage
        )
