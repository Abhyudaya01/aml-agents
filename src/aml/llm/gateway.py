"""LLMGateway: the only way agents reach a model.

Every call: mask inputs -> render -> raw-PII check -> provider (validate/retry) ->
audit event -> return output (still tokenized). Failures are audited too.

Agents receive a gateway, never a provider. tests/test_llm_boundary.py fails if
agent-side code imports provider modules or LLM SDKs, or reaches into the
gateway's private attributes.
"""

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from aml.audit.log import AuditLog
from aml.llm.pricing import ModelPrice, cost_usd
from aml.llm.providers.base import Provider, generate_structured
from aml.llm.types import LLMError, LLMResult, Prompt, StructuredOutputError, T
from aml.pii.tokenizer import PIIDetector, PIILeakError, PIIMasker


class PromptSpec(BaseModel):
    """What an agent hands the gateway: fixed instructions plus structured context.

    `context` is evidence objects from tools; it is masked before rendering.
    """

    system: str
    instructions: str
    context: dict[str, Any] = {}


def render(system: str, instructions: str, masked_context: Any) -> Prompt:
    body = json.dumps(masked_context, indent=2, sort_keys=True, default=str)
    return Prompt(system=system, user=f"{instructions}\n\n<context>\n{body}\n</context>")


def prompt_hash(prompt: Prompt) -> str:
    return hashlib.sha256(f"{prompt.system}\x00{prompt.user}".encode()).hexdigest()


class LLMGateway:
    def __init__(
        self,
        provider: Provider,
        masker: PIIMasker,
        audit: AuditLog,
        price: ModelPrice,
        max_retries: int = 2,
        detector: PIIDetector | None = None,
    ) -> None:
        self._provider = provider
        self._masker = masker
        self._audit = audit
        self._price = price
        self._max_retries = max_retries
        self._detector = detector

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    def generate(self, case_id: str, actor: str, spec: PromptSpec, schema: type[T]) -> LLMResult[T]:
        prompt = render(
            self._masker.mask_text(spec.system),
            self._masker.mask_text(spec.instructions),
            self._masker.mask(spec.context),
        )
        p_hash = prompt_hash(prompt)
        base = {"schema": schema.__name__, "system": prompt.system, "user": prompt.user}

        leaks = self._detector.find(prompt.system + prompt.user) if self._detector else []
        if leaks:
            # Log that a leak was blocked, never the leaked values themselves.
            self._audit.append(
                case_id, actor, "llm_call_blocked",
                {"schema": schema.__name__, "reason": "raw_pii", "n_hits": len(leaks)},
                model_id=self.model_id, prompt_hash=p_hash,
            )  # fmt: skip
            raise PIILeakError(f"{len(leaks)} raw PII value(s) in rendered prompt")

        try:
            result = generate_structured(
                self._provider, prompt, schema, self._price, self._max_retries
            )
        except StructuredOutputError as e:
            self._audit.append(
                case_id, actor, "llm_call_failed",
                {**base, "error": "validation", "attempts": e.attempts,
                 "response": self._masker.mask_text(e.text)},
                model_id=self.model_id, prompt_hash=p_hash,
                tokens={"input": e.usage.input_tokens, "output": e.usage.output_tokens},
                cost=cost_usd(e.usage, self._price), latency=e.latency_ms,
            )  # fmt: skip
            raise
        except LLMError as e:
            self._audit.append(
                case_id, actor, "llm_call_failed",
                {**base, "error": type(e).__name__, "detail": str(e)},
                model_id=self.model_id, prompt_hash=p_hash,
            )  # fmt: skip
            raise

        self._audit.append(
            case_id, actor, "llm_call",
            {**base, "attempts": result.attempts,
             "response": self._masker.mask(result.output.model_dump(mode="json"))},
            model_id=result.model_id, prompt_hash=p_hash,
            tokens={"input": result.usage.input_tokens, "output": result.usage.output_tokens},
            cost=result.cost_usd, latency=result.latency_ms,
        )  # fmt: skip
        return result
