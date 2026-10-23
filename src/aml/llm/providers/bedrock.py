"""Amazon Bedrock via langchain-aws ChatBedrockConverse.

`with_structured_output(..., include_raw=True)` returns the raw AIMessage alongside
the parse, so usage survives a parse failure and the raw tool arguments can be fed
back for a retry.
"""

import json
from typing import Any

from pydantic import BaseModel

from aml.llm.providers.base import Provider
from aml.llm.types import Completion, Prompt, Usage


class BedrockProvider(Provider):
    def __init__(
        self, model_id: str, region: str, max_tokens: int, chat_model: Any | None = None
    ) -> None:
        if chat_model is None:
            from langchain_aws import ChatBedrockConverse

            chat_model = ChatBedrockConverse(
                model_id=model_id, region_name=region, max_tokens=max_tokens
            )
        self._chat = chat_model
        self.model_id = model_id

    def complete(self, prompt: Prompt, schema: type[BaseModel]) -> Completion:
        result = self._chat.with_structured_output(schema, include_raw=True).invoke(
            [("system", prompt.system), ("human", prompt.user)]
        )
        raw = result["raw"]
        if result.get("parsed") is not None:
            text = result["parsed"].model_dump_json()
        elif raw.tool_calls:
            text = json.dumps(raw.tool_calls[0].get("args", {}))
        else:
            text = raw.content if isinstance(raw.content, str) else json.dumps(raw.content)
        meta = raw.usage_metadata or {}
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", 0),
            ),
            model_id=self.model_id,
            stop_reason=(raw.response_metadata or {}).get("stopReason"),
        )
