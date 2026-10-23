"""Shared types for LLM calls."""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Prompt(BaseModel):
    """A fully rendered prompt. By the time a Prompt exists it is already masked."""

    system: str
    user: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Completion(BaseModel):
    """One raw provider response, before schema validation."""

    text: str
    usage: Usage
    model_id: str
    stop_reason: str | None = None


class LLMResult(BaseModel, Generic[T]):
    output: T
    raw_text: str
    usage: Usage
    latency_ms: float
    model_id: str
    cost_usd: float
    attempts: int


class LLMError(Exception):
    """Base class for LLM-layer failures."""


class LLMRefusalError(LLMError):
    """The model (and any fallback) declined the request. Not retried."""


class StructuredOutputError(LLMError):
    """The response never validated against the schema within the retry budget."""

    def __init__(self, message: str, usage: Usage, latency_ms: float, attempts: int, text: str):
        super().__init__(message)
        self.usage = usage
        self.latency_ms = latency_ms
        self.attempts = attempts
        self.text = text
