"""Safe application logging utilities."""

import logging
import os
SENSITIVE_ENVIRONMENT_KEYS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OLLAMA_API_KEY",
}

def configure_logging(level: str = "INFO") -> logging.Logger:
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    return logging.getLogger("technical-ai-patterns-labs")


def redact_environment(values: dict[str, str]) -> dict[str, str]:
    """Return environment values with known secret keys redacted."""

    redacted: dict[str, str] = {}

    for key, value in values.items():
        if key.upper() in SENSITIVE_ENVIRONMENT_KEYS:
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value

    return redacted