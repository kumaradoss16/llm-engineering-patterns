from decimal import Decimal

import pytest

from app.utilities.cost import ModelPricing, estimate_cost


def test_zero_tokens_cost_zero() -> None:
    pricing = ModelPricing(
        input_per_million=Decimal("0.15"),
        output_per_million=Decimal("0.60"),
    )

    result = estimate_cost(0, 0, pricing)

    assert result is not None
    assert result.total_cost == Decimal("0")


def test_missing_usage_returns_none() -> None:
    pricing = ModelPricing(
        input_per_million=Decimal("0.15"),
        output_per_million=Decimal("0.60"),
    )

    assert estimate_cost(None, None, pricing) is None


def test_missing_pricing_returns_none() -> None:
    assert estimate_cost(100, 200, None) is None


def test_negative_input_tokens_raise_error() -> None:
    pricing = ModelPricing(
        input_per_million=Decimal("0.15"),
        output_per_million=Decimal("0.60"),
    )

    with pytest.raises(ValueError):
        estimate_cost(-1, 10, pricing)