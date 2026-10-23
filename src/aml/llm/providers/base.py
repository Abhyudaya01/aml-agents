"""Provider interface plus the shared validate-and-retry loop."""

import json
import time
from abc import ABC, abstractmethod

from pydantic import BaseModel, ValidationError

from aml.llm.pricing import ModelPrice, cost_usd
from aml.llm.types import Completion, LLMResult, Prompt, StructuredOutputError, T, Usage


class Provider(ABC):
    """Returns the raw JSON text a model produced for `schema`; never validates it."""

    model_id: str

    @abstractmethod
    def complete(self, prompt: Prompt, schema: type[BaseModel]) -> Completion: ...


def _feedback(prompt: Prompt, bad_text: str, error: ValidationError) -> Prompt:
    errors = json.dumps(error.errors(include_url=False, include_input=False), default=str)
    return Prompt(
        system=prompt.system,
        user=(
            f"{prompt.user}\n\nYour previous response was:\n{bad_text}\n\n"
            f"It failed schema validation with these errors:\n{errors}\n"
            "Return a corrected JSON object that matches the schema exactly."
        ),
    )


def generate_structured(
    provider: Provider,
    prompt: Prompt,
    schema: type[T],
    price: ModelPrice,
    max_retries: int,
) -> LLMResult[T]:
    """Call the provider until the output validates, up to 1 + max_retries attempts.

    Usage, latency and cost accumulate across attempts, failed ones included.
    """
    usage, latency_ms, text = Usage(), 0.0, ""
    current = prompt
    for attempt in range(1, max_retries + 2):
        t0 = time.perf_counter()
        completion = provider.complete(current, schema)
        latency_ms += (time.perf_counter() - t0) * 1000
        usage += completion.usage
        text = completion.text
        try:
            output = schema.model_validate_json(text)
        except ValidationError as e:
            if attempt > max_retries:
                raise StructuredOutputError(
                    f"{schema.__name__} failed validation after {attempt} attempts: {e}",
                    usage=usage,
                    latency_ms=latency_ms,
                    attempts=attempt,
                    text=text,
                ) from e
            current = _feedback(prompt, text, e)
            continue
        return LLMResult[schema](
            output=output,
            raw_text=text,
            usage=usage,
            latency_ms=latency_ms,
            model_id=completion.model_id,
            cost_usd=cost_usd(usage, price),
            attempts=attempt,
        )
    raise AssertionError("unreachable")
