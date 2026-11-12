"""LLM-based technical summarization pattern."""

import json

from pydantic import ValidationError

from app.providers.base import BaseModelProvider, ModelResponse
from app.schemas.summarization import TechnicalSummary

class SummarizationError(Exception):
    """Raised when summarization cannot produce a valid result."""

MAX_SOURCE_CHARACTERS = 20_000

# Constructs the prompt that will be sent to the LLM.
def build_summary_prompt(
        text: str,
        style: str = "three action-oriented bullet points",
) -> str:
    return (
        f"Summarize the following technical content as {style}.\n\n"
        "Rules:\n"
        "- Use only information present in the source text.\n"
        "- Do not add assumptions, recommendations, or facts not in the source.\n"
        "- Preserve important services, error codes, versions, ports, and actions.\n"
        "- Return only valid JSON.\n"
        "- Use this exact JSON shape:\n"
        '{"summary": "<one-paragraph summary>", '
        '"key_points": ["<point 1>", "<point 2>", "<point 3>"]}\n\n'
        f"Source text:\n{text}"
    )


def summarize_text(
        provider: BaseModelProvider,
        text: str,
        style: str = "three action-oriented bullet points",
) -> tuple[TechnicalSummary, ModelResponse]:
    cleaned = text.strip()

    if not cleaned:
        raise SummarizationError("Source text cannot be empty.")

    if len(cleaned) > MAX_SOURCE_CHARACTERS:
        raise SummarizationError(
            f"Source text exceeds the {MAX_SOURCE_CHARACTERS}-character limit. "
            "Use chunked summarization or retrieval for longer documents."
        )
    # Generate the prompt and call the LLM
    prompt = build_summary_prompt(cleaned, style)
    response = provider.generate(prompt, temperature=0.2)

    if response.error:
        raise SummarizationError(f"Model request failed: {response.error}")
    if not response.text:
        raise SummarizationError("Model returned an empty summary.")

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise SummarizationError(
            f"Model returned invalid JSON: {response.text!r}"
        ) from exc

    try:
        result = TechnicalSummary.model_validate(payload)
    except ValidationError as exc:
        raise SummarizationError(
            f"Model summary failed schema validation: {exc}"
        ) from exc

    return result, response



