import pytest
from pydantic import ValidationError

from app.patterns.tool_calling import (
    ToolCallError,
    convert_temperature,
    execute_tool,
    get_service_status,
    read_approved_record,
    run_tool_call,
    search_local_knowledge,
)
from app.providers.base import BaseModelProvider, ModelResponse
from app.schemas.tool_calling import ToolRequest


def test_get_service_status_known_service() -> None:
    result = get_service_status("payments-service")

    assert result["status"] == "running"
    assert result["healthy"] is True


def test_get_service_status_payments_alias() -> None:
    result = get_service_status("payments")

    assert result["status"] == "running"
    assert result["healthy"] is True


def test_get_service_status_unknown_service() -> None:
    result = get_service_status("unknown-service")

    assert result["status"] == "unknown"
    assert result["healthy"] is False


def test_read_approved_record_success() -> None:
    result = read_approved_record("nginx-restart")

    assert result["title"] == "Restarting nginx"


def test_read_unapproved_record_raises_error() -> None:
    with pytest.raises(ToolCallError):
        read_approved_record("secret-record")


def test_temperature_conversion_celsius_to_fahrenheit() -> None:
    result = convert_temperature(25.0, "celsius")

    assert result["output_value"] == 77.0
    assert result["output_unit"] == "fahrenheit"


def test_invalid_temperature_unit_raises_error() -> None:
    with pytest.raises(ToolCallError):
        convert_temperature(25.0, "kelvin")


def test_tool_request_rejects_unknown_tool() -> None:
    with pytest.raises(ValidationError):
        ToolRequest(
            tool="delete_database",
            arguments={},
        )


def test_execute_tool_service_status() -> None:
    request = ToolRequest(
        tool="get_service_status",
        arguments={"service": "nginx"},
    )

    result = execute_tool(request)

    assert result["service"] == "nginx"


def test_execute_tool_missing_argument_raises_error() -> None:
    request = ToolRequest(
        tool="get_service_status",
        arguments={},
    )

    with pytest.raises(ToolCallError):
        execute_tool(request)


def test_local_search_finds_matching_document() -> None:
    result = search_local_knowledge("restart nginx")

    assert result["matched_document_id"] == "nginx-restart"


def test_run_tool_call_handles_markdown_code_fences() -> None:
    class MockProvider(BaseModelProvider):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
            self.calls.append(prompt)
            if len(self.calls) == 1:
                # Returns tool request with trailing markdown backticks
                return ModelResponse(
                    text='{"tool": "get_service_status", "arguments": {"service": "payments"}}\n```',
                    latency_ms=10.0,
                    input_tokens=50,
                    output_tokens=20,
                )
            # Returns final answer enclosed in markdown code fences
            return ModelResponse(
                text='```json\n{"answer": "Payments service is running.", "sources": ["tool:get_service_status"], "confidence": "high"}\n```',
                latency_ms=12.0,
                input_tokens=60,
                output_tokens=25,
            )

    provider = MockProvider()
    answer, selection, final, tool_request, tool_result = run_tool_call(
        provider, "What is the current status of the payments service?"
    )

    assert answer == "Payments service is running."
    assert tool_request.tool == "get_service_status"
    assert tool_request.arguments == {"service": "payments"}
    assert tool_result["status"] == "running"
    assert tool_result["healthy"] is True


def test_run_tool_call_invalid_json_raises_tool_call_error() -> None:
    class InvalidJsonProvider(BaseModelProvider):
        def generate(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
            return ModelResponse(
                text="I cannot help with this.",
                latency_ms=10.0,
                input_tokens=10,
                output_tokens=10,
            )

    with pytest.raises(ToolCallError):
        run_tool_call(InvalidJsonProvider(), "What is the current status of the payments service?")
