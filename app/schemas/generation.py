"""Pydantic schemas for generated technical content."""

from pydantic import BaseModel, Field

class TechnicalExplanation(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    explanation: str = Field(min_length=1, max_length=5_000)
    diagnostic_steps: list[str] = Field(default_factory=list, max_length=10)
    cautions: list[str] = Field(default_factory=list, max_length=10)


class GeneratedReport(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=5_000)
    findings: list[str] = Field(default_factory=list, max_length=20)
    next_steps: list[str] = Field(default_factory=list, max_length=20)
