import pytest
from pydantic import ValidationError

from app.patterns.generation import (
    GenerationError,
    build_explanation_prompt,
    generate_explanation,
    generate_report,
    validate_generated_content,
)
from app.providers.base import ModelResponse
from app.schemas.generation import TechnicalExplanation


def test_valid_explanation_schema() -> None:
    result = TechnicalExplanation.model_validate(
        {
            "title": "HTTP 502",
            "explanation": "The proxy could not reach the upstream service.",
            "diagnostic_steps": ["Check service status"],
            "cautions": ["Review impact before restarting"],
        }
    )

    assert result.title == "HTTP 502"


def test_empty_title_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalExplanation(
            title="",
            explanation="Valid explanation",
        )


def test_forbidden_command_is_rejected() -> None:
    with pytest.raises(GenerationError):
        validate_generated_content("Run this command: rm -rf /")


def test_prompt_includes_context() -> None:
    prompt = build_explanation_prompt(
        "Why did the deployment fail?",
        context="The PostgreSQL migration timed out.",
    )

    assert "The PostgreSQL migration timed out." in prompt


def test_empty_question_raises_error() -> None:
    provider = object()

    with pytest.raises(GenerationError):
        generate_explanation(provider, "   ")  # type: ignore[arg-type]


def test_report_requires_findings() -> None:
    provider = object()

    with pytest.raises(GenerationError):
        generate_report(provider, [])  # type: ignore[arg-type]