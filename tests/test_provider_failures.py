import pytest

from app.patterns.classification import (
    ClassificationError,
    _parse_model_response,
)
from app.providers.base import ModelResponse
from app.schemas.classification import TicketClassification


def test_timeout_is_raised_as_classification_error() -> None:
    response = ModelResponse(
        text="",
        latency_ms=0.0,
        input_tokens=None,
        output_tokens=None,
        error="timeout",
    )

    with pytest.raises(ClassificationError):
        _parse_model_response(response, TicketClassification)


def test_connection_error_is_raised_as_classification_error() -> None:
    response = ModelResponse(
        text="",
        latency_ms=0.0,
        input_tokens=None,
        output_tokens=None,
        error="connection_error",
    )

    with pytest.raises(ClassificationError):
        _parse_model_response(response, TicketClassification)


def test_rate_limit_is_raised_as_classification_error() -> None:
    response = ModelResponse(
        text="",
        latency_ms=0.0,
        input_tokens=None,
        output_tokens=None,
        error="rate_limited",
    )

    with pytest.raises(ClassificationError):
        _parse_model_response(response, TicketClassification)