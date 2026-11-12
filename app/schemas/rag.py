"""Pydantic schemas for basic RAG."""

from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    chunk_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    embedding: list[float]


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    text: str
    score: float


class RagAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=5_000)
    sources: list[str] = Field(default_factory=list, max_length=10)
    confidence: str = Field(pattern="^(high|medium|low)$")