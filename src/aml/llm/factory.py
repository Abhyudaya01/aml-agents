"""Builds the configured provider and wraps it in an LLMGateway.

This is the one place outside tests that constructs providers.
"""

from aml.audit.log import AuditLog
from aml.config import Settings
from aml.llm.gateway import LLMGateway
from aml.llm.pricing import ModelPrice, resolve_price
from aml.llm.providers.base import Provider
from aml.pii.tokenizer import PIIDetector, PIIMasker

DEFAULT_MODELS = {"fake": "fake", "anthropic": "claude-opus-5-5"}


def price_table(settings: Settings) -> dict[str, ModelPrice]:
    merged = {**settings.llm_prices, **settings.llm_price_overrides}
    return {k: ModelPrice(**v) for k, v in merged.items()}


def resolve_model_id(settings: Settings) -> str:
    model_id = settings.model_id or DEFAULT_MODELS.get(settings.llm_provider)
    if not model_id:
        raise ValueError(f"AML_MODEL_ID must be set for provider {settings.llm_provider!r}")
    return model_id


def make_provider(settings: Settings) -> Provider:
    model_id = resolve_model_id(settings)
    if settings.llm_provider == "fake":
        from aml.llm.providers.fake import FakeLLM

        return FakeLLM(script=[], model_id=model_id)
    if settings.llm_provider == "anthropic":
        from aml.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model_id, settings.llm_effort, settings.llm_max_tokens)
    if settings.llm_provider == "openai":
        from aml.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(model_id, settings.llm_max_tokens)
    if settings.llm_provider == "bedrock":
        from aml.llm.providers.bedrock import BedrockProvider

        if not settings.aws_region:
            raise ValueError("AML_AWS_REGION must be set for the bedrock provider")
        return BedrockProvider(model_id, settings.aws_region, settings.llm_max_tokens)
    raise ValueError(f"unknown provider {settings.llm_provider!r}")


def make_gateway(
    settings: Settings,
    masker: PIIMasker,
    audit: AuditLog,
    detector: PIIDetector | None = None,
    provider: Provider | None = None,
) -> LLMGateway:
    """Price is resolved before any call, so an unpriced model fails fast."""
    provider = provider or make_provider(settings)
    price = resolve_price(provider.model_id, price_table(settings))
    return LLMGateway(provider, masker, audit, price, settings.llm_max_retries, detector)
