"""Claude via the official Anthropic SDK.

Structured output uses `messages.parse(output_format=Schema)`, which constrains the
response to the schema. We read the raw HTTP body ourselves (text + usage) rather
than the SDK's parsed object, so that a response that still fails our validation
keeps its token usage and can be fed back for a retry.

Server-side refusal fallbacks are enabled (`fallbacks="default"`): if the model
declines on policy grounds, the API re-runs the request on a fallback model in the
same call. A final `refusal` stop reason raises LLMRefusalError.
"""

from typing import Any

from pydantic import BaseModel

from aml.llm.providers.base import Provider
from aml.llm.types import Completion, LLMRefusalError, Prompt, Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(Provider):
    def __init__(
        self, model_id: str, effort: str, max_tokens: int, client: Any | None = None
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.model_id = model_id
        self._effort = effort
        self._max_tokens = max_tokens

    def complete(self, prompt: Prompt, schema: type[BaseModel]) -> Completion:
        raw = self._client.beta.messages.with_raw_response.parse(
            model=self.model_id,
            max_tokens=self._max_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
            output_format=schema,
            output_config={"effort": self._effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        body = raw.http_response.json()
        if body.get("stop_reason") == "refusal":
            details = body.get("stop_details") or {}
            raise LLMRefusalError(f"refused (category={details.get('category')})")
        text = "".join(b["text"] for b in body["content"] if b.get("type") == "text")
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            model_id=body.get("model", self.model_id),
            stop_reason=body.get("stop_reason"),
        )
