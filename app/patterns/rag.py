"""Minimal retrieval-augmented generation implementation.

This is a Phase 1 educational demonstration. It uses an in-memory knowledge
base and simple fixed-size chunking. Production RAG requires more advanced
ingestion, indexing, retrieval, evaluation, and security controls.
"""

import json
from dataclasses import dataclass

from pydantic import ValidationError

from app.patterns.similarity import cosine_similarity
from app.providers.base import BaseModelProvider, ModelResponse
from app.providers.embeddings import EmbeddingError, OllamaEmbeddingProvider
from app.schemas.rag import DocumentChunk, RagAnswer, RetrievedChunk


class RagError(Exception):
    """Raised when RAG cannot produce a valid grounded response."""


MAX_QUESTIONS_CHARACTERS = 2_000
MAX_CONTENT_CHARACTERS = 8_000
DEFAULT_TOP_K = 2
DEFAULT_MINIMUM_SCORE = 0.0

@dataclass
class InMemoryKnowledgeBase:
    chunks: list[DocumentChunk]

    def add_document(
            self,
            document_id: str,
            title: str,
            text: str,
            embedding_provider: OllamaEmbeddingProvider,
            chunk_sizes: int = 800,
            overlap: int = 100,
    ) -> int:
        """Chunk a document, embed chunk, and store it in memory."""

        cleaned = text.strip()

        if not cleaned:
            raise RagError("Document text cannot be empty.")

        if chunk_sizes <= overlap:
            raise RagError("Chunk size must be greater than overlap.")

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_number = 0

        while start < len(cleaned):
            end = min(start + chunk_sizes, len(cleaned))
            chunk_text = cleaned[start:end].strip()

            if chunk_text:
                embedding = embedding_provider.create_embedding(chunk_text)
                chunk_number += 1


                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}-chunk-{chunk_number}",
                        title=title,
                        text=chunk_text,
                        embedding=embedding,
                    )
                )

            if end >= len(cleaned):
                break

            start += chunk_sizes - overlap
        self.chunks.extend(chunks)
        return len(chunks)


    def retrieve(
            self,
            question: str,
            embedding_provider: OllamaEmbeddingProvider,
            top_k: int = DEFAULT_TOP_K,
            minimum_score: float = DEFAULT_MINIMUM_SCORE,
    ) -> list[RetrievedChunk]:
        """Retrieve the most semantically similar chunks."""

        if not self.chunks:
            raise RagError("Knowledge base is empty.")

        if not question.strip():
            raise RagError("Question cannot be empty.")

        if len(question) > MAX_CONTENT_CHARACTERS:
            raise RagError("Question exceeds the allowed length.")

        query_embedding = embedding_provider.create_embedding(question)   # embedding model converts the question into a vector.
        scored_chunks: list[RetrievedChunk] = []

        for chunk in self.chunks:
            score = cosine_similarity(query_embedding, chunk.embedding)

            if score >= minimum_score:
                scored_chunks.append(
                    RetrievedChunk(
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        title=chunk.title,
                        text=chunk.text,
                        score=round(score, 6),
                    )
                )

        scored_chunks.sort(key=lambda item: item.score, reverse=True)
        return scored_chunks[:top_k]

# Convert the retrieved document chunks into a structured prompt that the generative model can understand.
def build_rag_prompt(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    context_parts: list[str] = []
    total_characters = 0

    for index, chunk in enumerate(retrieved_chunks, start=1):
        source_label = f"[{index}] {chunk.title} ({chunk.document_id})"
        chunk_block = f"{source_label}\n{chunk.text}\n"

        if total_characters + len(chunk_block) > MAX_CONTENT_CHARACTERS:
            break

        context_parts.append(chunk_block)
        total_characters += len(chunk_block)

        if not context_parts:
            raise RagError("No retrieved context file within the context budget.")

        context = "\n".join(context_parts)

        return (
            "Answer the user question using only the provided context.\n\n"
            "Rules:\n"
            "- Do not use knowledge outside the provided context.\n"
            "- If the context does not contain enough information, say that the "
            "answer is not available in the provided sources.\n"
            "- Do not invent commands, configuration values, or procedures.\n"
            "- Return only valid JSON.\n"
            "- Use this exact JSON shape:\n"
            "{\n"
            '  "answer": "<answer>",\n'
            '  "sources": ["[1]"],\n'
            '  "confidence": "high|medium|low"\n'
            "}\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )

# Connects the retrieval process with the LLM generation process.
def answer_question(
        generator: BaseModelProvider,
        knowledge_base: InMemoryKnowledgeBase,
        embedding_provider: OllamaEmbeddingProvider,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        minimum_score: float = DEFAULT_MINIMUM_SCORE,
) -> tuple[RagAnswer, ModelResponse, list[RetrievedChunk]]:
    """Retrieve context and generate a grounded answer."""

    retrieved = knowledge_base.retrieve(
        question=question,
        embedding_provider=embedding_provider,
        top_k=top_k,
        minimum_score=minimum_score,
    )

    if not retrieved:
        return RagAnswer(
            answer="The answer is not available in the provided knowledge base.",
            sources=[],
            confidence="low",
        ), ModelResponse(
            text="",
            latency_ms=0.0,
            input_tokens=None,
            output_tokens=None,
        ), []

    prompt = build_rag_prompt(question, retrieved)
    response = generator.generate(prompt, temperature=0.0)

    if response.error:
        raise RagError(f"Model request failed: {response.error}")

    if not response.text:
        raise RagError("Model returned an empty response.")

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RagError(
            f"Model returned invalid JSON: {response.text!r}"
        ) from exc

    try:
        result = RagAnswer.model_validate(payload)
    except ValidationError as exc:
        raise RagError(
            f"RAG response failed schema validation: {exc}"
        ) from exc

    return result, response, retrieved









