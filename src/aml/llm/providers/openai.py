"""OpenAI via the official SDK's Responses API with a pydantic `text_format`.

As with Anthropic, the raw body is read directly so failed validations keep usage.
"""

from typing import Any

from pydantic import BaseModel

from aml.llm.providers.base import Provider
from aml.llm.types import Completion, LLMRefusalError, Prompt, Usage


class OpenAIProvider(Provider):
    def __init__(self, model_id: str, max_tokens: int, client: Any | None = None) -> None:
        if client is None:
            import openai

            client = openai.OpenAI()
        self._client = client
        self.model_id = model_id
        self._max_tokens = max_tokens

    def complete(self, prompt: Prompt, schema: type[BaseModel]) -> Completion:
        raw = self._client.responses.with_raw_response.parse(
            model=self.model_id,
            instructions=prompt.system,
            input=prompt.user,
            text_format=schema,
            max_output_tokens=self._max_tokens,
        )
        body = raw.http_response.json()
        parts = [
            c for item in body.get("output", []) if item.get("type") == "message"
            for c in item.get("content", [])
        ]  # fmt: skip
        if any(c.get("type") == "refusal" for c in parts):
            raise LLMRefusalError("refused")
        usage = body.get("usage") or {}
        return Completion(
            text="".join(c.get("text", "") for c in parts if c.get("type") == "output_text"),
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            model_id=body.get("model", self.model_id),
            stop_reason=body.get("status"),
        )
