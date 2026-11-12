import pytest
from pydantic import ValidationError

from app.patterns.classification import (
    ClassificationError,
    _parse_model_response,
)
from app.providers.base import ModelResponse
from app.schemas.classification import TicketClassification


def test_valid_ticket_response_passes() -> None:
    response = ModelResponse(
        text='{"label": "Network", "reasoning": "VPN issue"}',
        latency_ms=100.0,
        input_tokens=20,
        output_tokens=10,
    )

    result = _parse_model_response(response, TicketClassification)

    assert result.label == "Network"


def test_invalid_label_fails_validation() -> None:
    response = ModelResponse(
        text='{"label": "Database", "reasoning": "Wrong category"}',
        latency_ms=100.0,
        input_tokens=20,
        output_tokens=10,
    )

    with pytest.raises(ClassificationError):
        _parse_model_response(response, TicketClassification)


def test_invalid_json_fails() -> None:
    response = ModelResponse(
        text="The correct category is Network.",
        latency_ms=100.0,
        input_tokens=20,
        output_tokens=10,
    )

    with pytest.raises(ClassificationError):
        _parse_model_response(response, TicketClassification)


def test_provider_error_is_raised() -> None:
    response = ModelResponse(
        text="",
        latency_ms=0.0,
        input_tokens=None,
        output_tokens=None,
        error="timeout",
    )

    with pytest.raises(ClassificationError):
        _parse_model_response(response, TicketClassification)


def test_schema_rejects_unknown_label_directly() -> None:
    with pytest.raises(ValidationError):
        TicketClassification(label="Unknown")