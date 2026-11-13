"""Pydantic schemas for summarization outputs."""

from pydantic import BaseModel, Field

class TechnicalSummary(BaseModel):
    summary: str = Field(min_length=1, max_length=5_000)
    key_points: list[str] = Field(default_factory=list, max_length=10)