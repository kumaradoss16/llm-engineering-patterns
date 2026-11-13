import pytest
from pydantic import ValidationError

from app.patterns.summarization import (
    SummarizationError,
    build_summary_prompt,
    summarize_text,
)
from app.providers.base import ModelResponse
from app.schemas.summarization import TechnicalSummary


def test_valid_summary_schema() -> None:
    result = TechnicalSummary.model_validate(
        {
            "summary": "Database connectivity failed during deployment.",
            "key_points": ["Migration timed out", "Service recovered"],
        }
    )

    assert result.summary.startswith("Database")
    assert len(result.key_points) == 2


def test_empty_summary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalSummary(summary="", key_points=[])


def test_prompt_contains_grounding_rules() -> None:
    prompt = build_summary_prompt("The service returned HTTP 502.")

    assert "only information present in the source text" in prompt
    assert "The service returned HTTP 502." in prompt


def test_empty_input_raises_error() -> None:
    provider = object()

    with pytest.raises(SummarizationError):
        summarize_text(provider, "   ")  # type: ignore[arg-type]


def test_invalid_model_json_raises_error() -> None:
    class FakeProvider:
        def generate(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
            return ModelResponse(
                text="Here is the summary: the service failed.",
                latency_ms=100.0,
                input_tokens=20,
                output_tokens=10,
            )

    with pytest.raises(SummarizationError):
        summarize_text(FakeProvider(), "The service failed.")  # type: ignore[arg-type]