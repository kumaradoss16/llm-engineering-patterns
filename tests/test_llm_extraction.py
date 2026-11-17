import pytest
from pydantic import ValidationError

from app.patterns.llm_extraction import (
    LLMExtractionError,
    IPAddressExtraction,
    build_ip_extraction_prompt,
)


def test_valid_ip_address_passes_validation() -> None:
    result = IPAddressExtraction(ip_address="192.168.1.25")

    assert result.ip_address == "192.168.1.25"


def test_invalid_ip_address_fails_validation() -> None:
    with pytest.raises(ValidationError):
        IPAddressExtraction(ip_address="999.999.999.999")


def test_null_ip_address_is_allowed() -> None:
    result = IPAddressExtraction(ip_address=None)

    assert result.ip_address is None


def test_prompt_contains_json_instruction() -> None:
    prompt = build_ip_extraction_prompt("Connection failed to 192.168.1.25.")

    assert "Return only valid JSON" in prompt
    assert "192.168.1.25" in prompt


def test_empty_text_raises_error() -> None:
    provider = object()

    with pytest.raises(LLMExtractionError):
        from app.patterns.llm_extraction import extract_ip_with_llm

        extract_ip_with_llm(provider, "   ")  # type: ignore[arg-type]