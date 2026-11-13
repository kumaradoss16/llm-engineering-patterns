import pytest
from pydantic import ValidationError

from app.patterns.extraction import ExtractionError, _build_log_prompt
from app.schemas.extraction import LogExtraction


def test_valid_log_extraction() -> None:
    result = LogExtraction.model_validate(
        {
            "service": "nginx",
            "port": 80,
            "error_code": "EADDRINUSE",
            "ip_address": "0.0.0.0",
            "version": "1.24.0",
            "event": "port_bind_failure",
        }
    )

    assert result.service == "nginx"
    assert result.port == 80


def test_invalid_port_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LogExtraction(port=99999)


def test_invalid_ip_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LogExtraction(ip_address="999.999.999.999")


def test_missing_fields_default_to_none() -> None:
    result = LogExtraction()

    assert result.service is None
    assert result.port is None


def test_prompt_contains_json_instruction() -> None:
    prompt = _build_log_prompt("nginx failed to bind port 80")

    assert "Return only valid JSON" in prompt
    assert "nginx failed to bind port 80" in prompt


def test_empty_text_raises_error() -> None:
    from app.patterns.extraction import extract_log_information
    from app.providers.ollama import OllamaProvider

    with pytest.raises(ExtractionError):
        extract_log_information(OllamaProvider(), "   ")