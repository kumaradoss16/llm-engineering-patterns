"""Bounded, safe tool-calling fundamentals.

The LLM may request a tool, but only this application executes it. Tools are
read-only or safely bounded. No shell commands, file writes, database writes,
or network calls are permitted in this Phase 1 implementation."""

import json
from typing import Any

from pydantic import ValidationError

from app.providers.base import BaseModelProvider, ModelResponse
from app.schemas.tool_calling import (
    ApprovedRecord,
    KnowledgeSearchResult,
    ServiceStatus,
    TemperatureConversion,
    ToolRequest,
)

class ToolCallError(Exception):
    """Raised when tool selection, validation, or execution fails."""


APPROVED_RECORDS: dict[str, dict[str, str]] = {
    "nginx-restart": {
        "title": "Restarting nginx",
        "content": (
            "Validate configuration with nginx -t, then restart using "
            "systemctl restart nginx."
        ),
    },
    "postgres-connectivity": {
        "title": "PostgreSQL connectivity",
        "content": (
            "Check service status, host, port, credentials, firewall rules, "
            "and network policy."
        ),
    },
}


LOCAL_KNOWLEDGE: list[dict[str, str]] = [
    {
        "id": "nginx-restart",
        "title": "Restarting nginx",
        "text": (
            "Use systemctl restart nginx to restart the nginx service. "
            "Validate configuration with nginx -t."
        ),
    },
    {
        "id": "postgres-connectivity",
        "title": "PostgreSQL connectivity",
        "text": (
            "Check PostgreSQL service status, host, port, credentials, "
            "firewall rules, and network policy."
        ),
    },
]


def get_service_status(service: str) -> dict[str, Any]:
    known_services = {
        "payments": "running",
        "payments-service": "running",
        "nginx": "running",
        "postgres": "running",
    }

    normalized = service.lower().strip()
    status = known_services.get(normalized)
    if status is None and (normalized.endswith(" service") or normalized.endswith("-service")):
        base = normalized.removesuffix(" service").removesuffix("-service").strip()
        status = known_services.get(base) or known_services.get(f"{base}-service")

    if status is None:
        return {
            "service": service,
            "status": "unknown",
            "healthy": False,
            "details": "Service is not in the approved monitoring list.",
        }

    return {
        "service": service,
        "status": status,
        "healthy": True,
        "details": "Simulates health status for Phase 1 demonstration.",
    }

def read_approved_record(record_id: str) -> dict[str, Any]:
    """Read an approved, non-sensitive technical record."""

    record = APPROVED_RECORDS.get(record_id.lower())

    if record is None:
        raise ToolCallError(f"Approved record not found: {record_id}")

    return {
        "record_id": record_id,
        "title": record["title"],
        "content": record["content"],
    }


def convert_temperature(value: float, input_unit: str) -> dict[str, Any]:
    """Convert Celsius to Fahrenheit or Fahrenheit to Celsius."""

    unit = input_unit.strip().lower()

    if unit not in ["celsius", "fahrenheit"]:
        raise ToolCallError(
            "Input unit must be 'celsius' or 'fahrenheit'."
        )

    if unit == "celsius":
        converted = (value * 9 / 5) + 32
        return {
            "input_value": value,
            "input_unit": "celsius",
            "output_value": round(converted, 2),
            "output_unit": "fahrenheit",
        }

    converted = (value - 32) * 5 / 9
    return {
        "input_value": value,
        "input_unit": "fahrenheit",
        "output_value": round(converted, 2),
        "output_unit": "celsius",
    }


# Backwards compatibility alias
conversion_temperature = convert_temperature


def search_local_knowledge(query: str) -> dict[str, Any]:
    """Perform a simple keyword search over approved local knowledge."""

    cleaned_query = query.strip().lower()

    if not cleaned_query:
        raise ToolCallError("Search query cannot be empty.")

    for document in LOCAL_KNOWLEDGE:
        if any(
            word in document["text"].lower()
            for word in cleaned_query.split()
        ):
            return {
                "query": query,
                "matched_document_id": document["id"],
                "title": document["title"],
                "excerpt": document["text"],
                "score": None,
            }

    return {
        "query": query,
        "matched_document_id": None,
        "title": None,
        "excerpt": None,
        "score": None,
    }


def execute_tool(tool_request: ToolRequest) -> dict[str, Any]:
    """Execute only an approved, bounded tool."""

    tool = tool_request.tool
    arguments = tool_request.arguments

    if tool == "get_service_status":
        service = arguments.get("service")

        if not isinstance(service, str) or not service.strip():
            raise ToolCallError("Tool 'service' arguments must be a non-empty string.")

        return get_service_status(service.strip())

    if tool == "read_approved_record":
        record_id = arguments.get("record_id")

        if not isinstance(record_id, str) or not record_id.strip():
            raise ToolCallError(
                "Tool 'record_id' arguments must be a non-empty string."
            )

        return read_approved_record(record_id.strip())

    if tool == "convert_temperature":
        value = arguments.get("value")
        unit = arguments.get("unit")

        if not isinstance(value, (int, float)):
            raise ToolCallError("Tool 'value' argument must be match.")

        if not isinstance(unit, str):
            raise ToolCallError("tool 'unit' argument must be a string.")

        return convert_temperature(float(value), unit)

    if tool == "search_local_knowledge":
        query = arguments.get("query")

        if not isinstance(query, str):
            raise ToolCallError("Tool 'query' argument must be a string.")

        return search_local_knowledge(query)

    raise ToolCallError(f"Unsupported tool: {tool!r}")


def _clean_json_text(text: str) -> str:
    """Strip markdown code fences and surrounding text from model output."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        else:
            cleaned = cleaned.lstrip("`")

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def _parse_json_payload(text: str) -> Any:
    """Parse JSON from model response, handling markdown formatting or fences."""
    cleaned = text.strip()

    # 1. Direct parse attempt
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. Markdown fence stripped attempt
    cleaned_fences = _clean_json_text(cleaned)
    try:
        return json.loads(cleaned_fences)
    except json.JSONDecodeError:
        pass

    # 3. Extract substring between first '{' and last '}'
    first_brace = cleaned_fences.find("{")
    last_brace = cleaned_fences.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned_fences[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback to raising JSONDecodeError on original text
    return json.loads(text)


def build_tools_selection_prompt(user_request: str) -> str:
    return (
        "You can request one approved tool to help answer the user request.\n\n"
        "Available tools:\n"
        "1. get_service_status\n"
        "   Arguments: {\"service\": \"<service name>\"}\n"
        "2. read_approved_record\n"
        "   Arguments: {\"record_id\": \"<record id>\"}\n"
        "3. convert_temperature\n"
        "   Arguments: {\"value\": <number>, \"unit\": \"celsius|fahrenheit\"}\n"
        "4. search_local_knowledge\n"
        "   Arguments: {\"query\": \"<search text>\"}\n\n"
        "Rules:\n"
        "- Choose only one tool.\n"
        "- Return only valid JSON.\n"
        "- Do not wrap JSON in markdown code blocks.\n"
        "- Use this exact JSON shape:\n"
        '{"tool": "<tool name>", "arguments": {}}\n'
        "- Do not execute commands yourself.\n"
        "- Do not invent tools.\n\n"
        f"User request:\n{user_request}"
    )


def build_final_answer_prompt(
        user_request: str,
        tool_request: ToolRequest,
        tool_result: dict[str, Any],
) -> str:
    return (
        "Answer the user request using the supplied tool result.\n\n"
        "Rules:\n"
        "- Use only the supplied tool result.\n"
        "- Do not invent additional system details.\n"
        "- If the result is incomplete, state what information is unavailable.\n"
        "- Return only valid JSON.\n"
        "- Do not wrap JSON in markdown code blocks.\n"
        "- Use this exact JSON shape:\n"
        "{\n"
        '  "answer": "<answer>",\n'
        '  "sources": ["tool:<tool name>"],\n'
        '  "confidence": "high|medium|low"\n'
        "}\n\n"
        f"User request:\n{user_request}\n\n"
        f"Tool requested:\n{tool_request.model_dump_json()}\n\n"
        f"Tool result:\n{json.dumps(tool_result, indent=2)}"
    )


def run_tool_call(
        provider: BaseModelProvider,
        user_request: str,
) -> tuple[str, ModelResponse, ModelResponse, ToolRequest, dict[str, Any]]:
    """Run the complete request -> tool -> result -> final answer lifecycle."""

    cleaned_request = user_request.strip()

    # Validate the user request
    if not cleaned_request:
        raise ToolCallError("User request cannot be empty.")

    if len(cleaned_request) > 2_000:
        raise ToolCallError("User request exceeds the 2,000 character limit.")

    # Ask the LLM to select a tool
    selection_response = provider.generate(
        build_tools_selection_prompt(cleaned_request),
        temperature=0.0,
    )

    # Check the LLM response
    if selection_response.error:
        raise ToolCallError(
            f"Tool-selection request failed: {selection_response.error}"
        )

    # Parse and validate the tool request
    try:
        selection_payload = _parse_json_payload(selection_response.text)
        tool_request = ToolRequest.model_validate(selection_payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ToolCallError(
            f"Model returned an invalid tool request: "
            f"{selection_response.text!r}"
        ) from exc

    # Execute the selected tool
    tool_result = execute_tool(tool_request)

    # Generate the final answer
    final_response = provider.generate(
        build_final_answer_prompt(cleaned_request, tool_request, tool_result),
        temperature=0.0,
    )

    if final_response.error:
        raise ToolCallError(
            f"Final answer request failed: {final_response.error}"
        )

    # Parse the final JSON
    try:
        final_payload = _parse_json_payload(final_response.text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ToolCallError(
            f"Model returned invalid final JSON: {final_response.text!r}"
        ) from exc

    if not isinstance(final_payload, dict):
        raise ToolCallError(
            f"Model returned invalid final JSON: {final_response.text!r}"
        )

    answer = final_payload.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        raise ToolCallError("Model returned an empty final answer.")

    return answer.strip(), selection_response, final_response, tool_request, tool_result
