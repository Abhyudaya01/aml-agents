"""FakeLLM: deterministic scripted responses for tests. Never touches the network."""

import math
from collections.abc import Callable

from pydantic import BaseModel

from aml.llm.providers.base import Provider
from aml.llm.types import Completion, Prompt, Usage

ScriptItem = str | BaseModel | Callable[[Prompt, type[BaseModel]], str | BaseModel]


def _tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


class FakeLLM(Provider):
    """Replays `script` in order, one item per call.

    An item may be a JSON string, a pydantic object (serialized), or a callable
    receiving (prompt, schema). Every prompt received is kept in `self.prompts`.
    Token counts are len(text) / 4, so tests can assert exact costs.
    """

    def __init__(self, script: list[ScriptItem], model_id: str = "fake") -> None:
        self.model_id = model_id
        self._script = list(script)
        self.prompts: list[Prompt] = []

    def complete(self, prompt: Prompt, schema: type[BaseModel]) -> Completion:
        self.prompts.append(prompt)
        if not self._script:
            raise RuntimeError("FakeLLM script exhausted")
        item = self._script.pop(0)
        if callable(item) and not isinstance(item, BaseModel):
            item = item(prompt, schema)
        text = item.model_dump_json() if isinstance(item, BaseModel) else item
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=_tokens(prompt.system + prompt.user), output_tokens=_tokens(text)
            ),
            model_id=self.model_id,
            stop_reason="end_turn",
        )
