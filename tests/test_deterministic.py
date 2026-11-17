import pytest

from app.deterministic.ip_extractor import (
    extract_first_ipv4,
    extract_ipv4_addresses,
)


def test_extracts_valid_ipv4_address() -> None:
    text = "Connection failed to 192.168.1.25:5432."

    result = extract_first_ipv4(text)

    assert result == "192.168.1.25"


def test_extracts_multiple_unique_addresses() -> None:
    text = "Primary: 10.0.0.1, secondary: 10.0.0.1, backup: 192.168.1.25."

    result = extract_ipv4_addresses(text)

    assert result == ["10.0.0.1", "192.168.1.25"]


def test_ignores_invalid_address_like_text() -> None:
    text = "The value 999.999.999.999 is not a valid IPv4 address."

    result = extract_ipv4_addresses(text)

    assert result == []


def test_returns_none_when_no_ip_exists() -> None:
    result = extract_first_ipv4("No IP address in this text.")

    assert result is None


def test_empty_text_returns_empty_list() -> None:
    assert extract_ipv4_addresses("") == []
    assert extract_ipv4_addresses("   ") == []