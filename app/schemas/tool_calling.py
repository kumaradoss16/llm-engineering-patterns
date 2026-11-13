"""Pydantic schemas for safe tool calling."""

from pydantic import BaseModel, Field, field_validator

class ToolRequest(BaseModel):
    tool: str = Field(min_length=1, max_length=100)
    arguments: dict[str, object] = Field(default_factory=dict)

    @field_validator("tool")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        allowed = [
            "get_service_status",
            "read_approved_record",
            "convert_temperature",
            "search_local_knowledge",
        ]

        if value not in allowed:
            raise ValueError(f"Tool is not allowed: {value!r}")

        return value

class ServiceStatus(BaseModel):
    service: str
    status: str
    healthy: bool
    details: str

class ApprovedRecord(BaseModel):
    record_id: str
    title: str
    content: str

class TemperatureConversion(BaseModel):
    input_value: float
    input_unit: str
    output_value: float
    output_unit: str

class KnowledgeSearchResult(BaseModel):
    query: str
    matched_document_id: str | None
    title: str | None
    excerpt: str | None
    score: float | None

