import pytest

from app.utilities.token_usage import build_token_usage


def test_total_tokens_calculated() -> None:
    usage = build_token_usage(100, 50)

    assert usage.total_tokens == 150


def test_missing_output_tokens_returns_none_total() -> None:
    usage = build_token_usage(100, None)

    assert usage.total_tokens is None


def test_negative_input_tokens_raise_error() -> None:
    with pytest.raises(ValueError):
        build_token_usage(-1, 10)


def test_negative_output_tokens_raise_error() -> None:
    with pytest.raises(ValueError):
        build_token_usage(10, -1)