"""LLM-based classification pattern with strict output validation."""

import json
from pydantic import ValidationError

from ..providers.base import BaseModelProvider, ModelResponse
from ..schemas.classification import ErrorClassification, TicketClassification

class ClassificationError(Exception):
    """Raised when classification cannot produce a valid result."""

# Creates the instructions that will be sent to the LLM.
def _build_ticket_prompt(text: str) -> str:
    return (
        "Classify the technical support ticket into exactly one category:\n"
        "Network, Hardware, Software, Security.\n\n"
        "Rules:\n"
        "- Return only valid JSON.\n"
        '- Use this exact JSON shape: {"label": "<label>", "reasoning": "<short reason>"}\n'
        "- Do not include Markdown, explanations outside JSON, or extra fields.\n\n"
        f"Ticket:\n{text}"
    )

def _build_error_prompt(text: str) -> str:
    return (
        "Classify the Linux or infrastructure error into exactly one category:\n"
        "Configuration, Connectivity, Resource, Permission, Unknown.\n\n"
        "Rules:\n"
        "- Return only valid JSON.\n"
        '- Use this exact JSON shape: {"category": "<category>", "reasoning": "<short reason>"}\n'
        "- Do not include Markdown or extra fields.\n\n"
        f"Error:\n{text}"
    )

def _parse_model_response(
        response: ModelResponse,
        schema_type: type[TicketClassification] | type[ErrorClassification],
    ) -> TicketClassification | ErrorClassification:
    if response.error:
        raise ClassificationError(f"Model request failed: {response.error}")

    if not response.text:
        raise ClassificationError("Model returned an empty responses.")

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ClassificationError(
            f"Model returned invalid JSON: {response.text!r}"
        ) from exc

    try:
        return schema_type.model_validate(payload)
    except ValidationError as exc:
        raise ClassificationError(
            f"Model output failed schema validation: {exc}"
        ) from exc

def classify_ticket(
        provider: BaseModelProvider,
        text: str,
    ) -> tuple[TicketClassification, ModelResponse]:
    """Classify a technical support ticket into one allowed label."""
    cleaned = text.strip()

    if not cleaned:
        raise ClassificationError("Ticket text cannot be empty.")
    if len(cleaned) > 5_000:
        raise ClassificationError("Ticket text exceeds the 5,000 character limit.")

    prompt = _build_ticket_prompt(cleaned)
    response = provider.generate(prompt, temperature=0.0)   # Send the prompt to the LLM
    result = _parse_model_response(response, TicketClassification)   # Parse and validate the response

    if not isinstance(result, TicketClassification):
        raise ClassificationError("Unexpected classification result type.")

    return result, response


def classify_error(
        provider: BaseModelProvider,
        text: str,
    ) -> tuple[ErrorClassification, ModelResponse]:
    """Classify a Linux or infrastructure error into one allowed category."""
    cleaned = text.strip()
    if not cleaned:
        raise ClassificationError("Error text cannot be empty.")
    if len(cleaned) > 5_000:
        raise ClassificationError("Error text exceeds the 5,000 character limit")

    prompt = _build_error_prompt(cleaned)
    response = provider.generate(prompt, temperature=0.0)
    result = _parse_model_response(response, ErrorClassification)

    if not isinstance(result, ErrorClassification):
        raise ClassificationError("Unexpected error classification result type.")

    return result, response
