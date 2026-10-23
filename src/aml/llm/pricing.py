"""Per-model token prices. The table itself lives in config (DEFAULT_LLM_PRICES plus
overrides); this module only looks prices up and computes cost.

A model with no price entry is rejected when the client is built, so cost is never
silently logged as zero.
"""

from pydantic import BaseModel

from aml.llm.types import Usage


class ModelPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


class UnpricedModelError(ValueError):
    pass


def resolve_price(model_id: str, prices: dict[str, ModelPrice]) -> ModelPrice:
    try:
        return prices[model_id]
    except KeyError:
        raise UnpricedModelError(
            f"No price for model {model_id!r}. Add it to AML_LLM_PRICE_OVERRIDES, e.g. "
            f'\'{{"{model_id}": {{"input_per_mtok": 1.0, "output_per_mtok": 5.0}}}}\''
        ) from None


def cost_usd(usage: Usage, price: ModelPrice) -> float:
    return (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000
