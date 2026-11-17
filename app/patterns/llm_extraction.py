"""LLM-based extraction for comparison with deterministic extraction."""

import json

from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError

from app.providers.base import BaseModelProvider, ModelResponse


class IPAddressExtraction(BaseModel):
    ip_address: str | None = Field(default=None, max_length=45)

    @field_validator("ip_address")
    @classmethod
    def validate_ip_address(cls, value: str | None) -> str | None:
        if value is None:
            return None

        import ipaddress

        try:
            ipaddress.ip_address(value)
        except ValidationError as exc:
            raise ValidationError(f"Invalid IP address: {value!r}") from exc

        return value

class LLMExtractionError(Exception):
    """Raise when LLM extraction cannot produce a valid result."""


def build_ip_extraction_prompt(text: str) -> str:
    return (
        "Extract the first IPv4 address from the following text.\n\n"
        "Rules:\n"
        "- Return only valid JSON.\n"
        '- Use this exact JSON shape: {"ip_address": "<ipv4 address or null>"}\n'
        "- Use null if no IPv4 address is present.\n"
        "- Do not invent an IP address.\n\n"
        f"Text:\n{text}"
    )

def extract_ip_with_llm(
        provider: BaseModelProvider,
        text: str,
) -> tuple[IPAddressExtraction, ModelResponse]:
    cleaned = text.strip()

    if not cleaned:
        raise LLMExtractionError("Input text cannot be empty.")

    prompt = build_ip_extraction_prompt(cleaned)
    response = provider.generate(prompt, temperature=0.0)   # Sends the prompt to the configured LLM.

    if response.error:
        raise LLMExtractionError(f"Model request failed: {response.error}")

    if not response.text:
        raise LLMExtractionError("Model returned an empty response.")

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise LLMExtractionError(
        f"Model returned invalid JSON: {response.text!r}"
    ) from exc

    try:
        result = IPAddressExtraction.model_validate(payload)
    except ValidationError as exc:
        raise LLMExtractionError(
            f"Model output failed schema validation: {exc}"
        ) from exc

    return result, response




