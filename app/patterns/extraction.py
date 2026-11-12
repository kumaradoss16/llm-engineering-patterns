"""LLM-based structured information extraction."""

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..providers.base import BaseModelProvider, ModelResponse
from ..schemas.extraction import LogExtraction, SoftwareExtraction

SchemaType = TypeVar("SchemaType", bound=BaseModel)

class ExtractionError(Exception):
    """Raised when extraction cannot produce a valid structured result."""


def _build_log_prompt(text: str) -> str:
    return (
        "Extract technical information from the log text.\n\n"
        "Return only valid JSON with these fields:\n"
        "{\n"
        '  "service": string or null,\n'
        '  "port": integer or null,\n'
        '  "error_code": string or null,\n'
        '  "ip_address": string or null,\n'
        '  "version": string or null,\n'
        '  "event": string or null\n'
        "}\n\n"
        "Rules:\n"
        "- Use null for unknown values.\n"
        "- Do not invent values.\n"
        "- Do not include Markdown or explanations outside JSON.\n\n"
        f"Log:\n{text}"
    )

def _build_software_prompt(text: str) -> str:
    return (
        "Extract software and configuration information.\n\n"
        "Return only valid JSON with these fields:\n"
        "{\n"
        '  "software_name": string or null,\n'
        '  "version": string or null,\n'
        '  "configuration_key": string or null,\n'
        '  "configuration_value": string or null\n'
        "}\n\n"
        "Rules:\n"
        "- Use null for unknown values.\n"
        "- Do not invent values.\n"
        "- Return only JSON.\n\n"
        f"Text:\n{text}"
    )


def extract_structured_data(
        provider: BaseModelProvider,
        prompt: str,
        schema_type: type[SchemaType],
    ) -> tuple[SchemaType, ModelResponse]:
    response = provider.generate(prompt, temperature=0.0)   # Calling the LLM

    if response.error:
        raise ExtractionError(f"Model request failed: {response.error}")

    if not response.text:
        raise ExtractionError("Model returned an empty response.")

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"Model returned invalid JSON: {response.text!r}"
        ) from exc

    try:
        validated = schema_type.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionError(
            f"Model output failed schema validation: {exc}"
        ) from exc

    return validated, response


# Provides a dedicated interface for log extraction
def extract_log_information(
        provider: BaseModelProvider,
        text: str,
    ) -> tuple[LogExtraction, ModelResponse]:
    cleaned = text.strip()

    if not cleaned:
        raise ExtractionError("Log text cannot be empty.")

    if len(cleaned) > 10_000:
        raise ExtractionError("Log text exceeds the 10,000-character limit.")

    prompt = _build_log_prompt(cleaned)
    return extract_structured_data(provider, prompt, LogExtraction)


def extract_software_information(
        provider: BaseModelProvider,
        text: str,
    ) -> tuple[SoftwareExtraction, ModelResponse]:
    cleaned = text.strip()

    if not cleaned:
        raise ExtractionError("Input text cannot be empty.")

    if len(cleaned) > 10_000:
        raise ExtractionError("Input text exceeds the 10,000-character limit.")

    prompt = _build_software_prompt(cleaned)
    return extract_structured_data(provider, prompt, SoftwareExtraction)
