"""LLM-based technical content generation with validation."""

import json

from pydantic import ValidationError
from app.providers.base import BaseModelProvider, ModelResponse
from app.schemas.generation import GeneratedReport, TechnicalExplanation

class GenerationError(Exception):
    """Raised when generation cannot produce a valid result."""

FORBIDDEN_COMMAND_HINTS = (
    "rm -rf /",
    "mkfs",
    "dd if=",
    "shutdown -h now",
    "drop database",
)


# Construct the instructions that will be sent to the LLM.
def build_explanation_prompt(
        question: str,
        context: str | None = None,
) -> str:
    context_block = ""
    if context:
        context_block = f"\nUse only the following approved context: \n{context}\n"

    return (
        "You are a technical assistant for infrastructure and software teams.\n"
        "Explain the user's technical question clearly and safely.\n\n"
        "Rules:\n"
        "- Do not invent facts, commands, version numbers, or product features.\n"
        "- If information is missing, state what additional information is needed.\n"
        "- Do not provide destructive commands.\n"
        "- Return only valid JSON.\n"
        "- Use this exact JSON shape:\n"
        "{\n"
        '  "title": "<short title>",\n'
        '  "explanation": "<clear explanation>",\n'
        '  "diagnostic_steps": ["<step 1>", "<step 2>"],\n'
        '  "cautions": ["<caution 1>"]\n'
        "}\n\n"
        f"{context_block}"
        f"Question:\n{question}"
    )

# Checks whether generated text contains any forbidden command hint.
def validate_generated_content(text: str) -> None:
    """Reject clearly dangerous executable guidance."""

    lowered = text.lower()

    for forbidden in FORBIDDEN_COMMAND_HINTS:
        if forbidden in lowered:
            raise GenerationError(
                f"Generated content contains a forbidden command hint: {forbidden!r}"
            )

def generate_explanation(
        provider: BaseModelProvider,
        question: str,
        context: str | None = None,
) -> tuple[TechnicalExplanation, ModelResponse]:
    """Generate and validate a technical explanation."""
    cleaned_question = question.strip()

    if not cleaned_question:
        raise GenerationError("Question cannot be emtpy.")

    if len(cleaned_question) > 5_000:
        raise GenerationError("Question exceeds the 5,000 character limits.")

    if context and len(context) > 20_000:
        raise GenerationError("Context exceeds the 20,000 character limits.")

    prompt = build_explanation_prompt(cleaned_question, context)   # Build the prompt
    response = provider.generate(prompt, temperature=0.2)   # Call the LLm

    if response.error:
        raise GenerationError(f"Model request failed: {response.error}")

    if not response.text:
        raise GenerationError("Model returned an empty response.")

    validate_generated_content(response.text)

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            f"Model returned invalid JSON: {response.text!r}"
        ) from exc

    try:
        result = TechnicalExplanation.model_validate(payload)
    except ValidationError as exc:
        raise GenerationError(
            f"Generated content failed schema validation: {exc}"
        ) from exc

    validate_generated_content(result.explanation)
    validate_generated_content("".join(result.diagnostic_steps))

    return result, response

def generate_report(
        provider: BaseModelProvider,
        findings: list[str],
) -> tuple[GeneratedReport, ModelResponse]:
    """Generated a structured technical report from supplied findings."""

    if not findings:
        raise GenerationError("At least one finding is required.")

    cleaned_findings = [finding.strip() for finding in findings if finding.strip()]

    if not cleaned_findings:
        raise GenerationError("Findings cannot be empty.")

    findings_text = "\n".join(f" - {finding}" for finding in cleaned_findings)

    prompt = (
        "Create a concise technical report from the following findings.\n\n"
        "Rules:\n"
        "- Use only the supplied findings.\n"
        "- Do not invent root causes or remediation steps.\n"
        "- Return only valid JSON.\n"
        "- Use this exact JSON shape:\n"
        "{\n"
        '  "title": "<report title>",\n'
        '  "summary": "<short summary>",\n'
        '  "findings": ["<finding 1>"],\n'
        '  "next_steps": ["<next step 1>"]\n'
        "}\n\n"
        f"Findings:\n{findings_text}"
    )

    response = provider.generate(prompt, temperature=0.2)

    if response.error:
        raise GenerationError(f"Model request failed: {response.error}")

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            f"Model returned invalid JSON: {response.text!r}"
        ) from exc

    try:
        result = GeneratedReport.model_validate(payload)
    except ValidationError as exc:
        raise GenerationError(
            f"Generated report failed schema validation: {exc}"
        ) from exc

    return result, response




