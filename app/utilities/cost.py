"""Cost estimation using Decimal.

Only calculate cost when bot token usage and verified pricing are available."""

from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True)
class ModelPricing:
    input_per_million: Decimal
    output_per_million: Decimal

@dataclass(frozen=True)
class CostEstimate:
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal

def estimate_cost(
        input_tokens: int | None,
        output_tokens: int | None,
        pricing: ModelPricing | None,
) -> CostEstimate | None:
    if pricing is None or input_tokens is None or output_tokens is None:
        return None

    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    input_cost = (
        Decimal(input_tokens) * pricing.input_per_million / Decimal(1_000_000)
    )

    output_cost = (
        Decimal(output_tokens) * pricing.output_per_million / Decimal(1_000_000)
    )

    return CostEstimate(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )