from app.utilities.logging import redact_environment


def test_api_key_is_redacted() -> None:
    values = {
        "OPENAI_API_KEY": "secret-key",
        "OLLAMA_BASE_URL": "http://localhost:11434",
    }

    result = redact_environment(values)

    assert result["OPENAI_API_KEY"] == "[REDACTED]"
    assert result["OLLAMA_BASE_URL"] == "http://localhost:11434"


def test_non_secret_values_are_preserved() -> None:
    values = {
        "OLLAMA_MODEL": "granite4.2:3b",
    }

    result = redact_environment(values)

    assert result["OLLAMA_MODEL"] == "granite4.2:3b"