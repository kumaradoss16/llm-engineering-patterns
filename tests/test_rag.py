import pytest

from app.patterns.rag import (
    InMemoryKnowledgeBase,
    RagError,
    build_rag_prompt,
)
from app.schemas.rag import DocumentChunk, RetrievedChunk


class FakeEmbeddingProvider:
    def create_embedding(self, text: str) -> list[float]:
        if "nginx" in text.lower():
            return [1.0, 0.0]
        if "postgres" in text.lower():
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_add_document_creates_chunks() -> None:
    knowledge_base = InMemoryKnowledgeBase(chunks=[])
    provider = FakeEmbeddingProvider()

    count = knowledge_base.add_document(
        document_id="nginx",
        title="Nginx",
        text="Restart nginx with systemctl. Reload configuration with systemctl reload nginx.",
        embedding_provider=provider,  # type: ignore[arg-type]
        chunk_size=50,
        overlap=10,
    )

    assert count > 0
    assert len(knowledge_base.chunks) == count


def test_empty_document_raises_error() -> None:
    knowledge_base = InMemoryKnowledgeBase(chunks=[])
    provider = FakeEmbeddingProvider()

    with pytest.raises(RagError):
        knowledge_base.add_document(
            document_id="empty",
            title="Empty",
            text="   ",
            embedding_provider=provider,  # type: ignore[arg-type]
        )


def test_invalid_chunk_configuration_raises_error() -> None:
    knowledge_base = InMemoryKnowledgeBase(chunks=[])
    provider = FakeEmbeddingProvider()

    with pytest.raises(RagError):
        knowledge_base.add_document(
            document_id="invalid",
            title="Invalid",
            text="Some text",
            embedding_provider=provider,  # type: ignore[arg-type]
            chunk_size=10,
            overlap=20,
        )


def test_retrieve_ranks_relevant_chunk_first() -> None:
    knowledge_base = InMemoryKnowledgeBase(
        chunks=[
            DocumentChunk(
                document_id="nginx",
                chunk_id="nginx-1",
                title="Nginx",
                text="Restart nginx with systemctl.",
                embedding=[1.0, 0.0],
            ),
            DocumentChunk(
                document_id="postgres",
                chunk_id="postgres-1",
                title="Postgres",
                text="Check PostgreSQL connectivity.",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    results = knowledge_base.retrieve(
        question="How do I restart nginx?",
        embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
    )

    assert results[0].document_id == "nginx"


def test_rag_prompt_contains_context_and_question() -> None:
    chunk = RetrievedChunk(
        document_id="nginx",
        chunk_id="nginx-1",
        title="Nginx",
        text="Use systemctl restart nginx.",
        score=0.9,
    )

    prompt = build_rag_prompt("How do I restart nginx?", [chunk])

    assert "Use systemctl restart nginx." in prompt
    assert "How do I restart nginx?" in prompt
    assert "using only the provided context" in prompt