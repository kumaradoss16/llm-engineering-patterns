"""Semantic similarity using embeddings and cosine similarity."""

from dataclasses import dataclass

from app.providers.embeddings import EmbeddingError, OllamaEmbeddingProvider

class SimilarityError(Exception):
    """Raised when semantic similarity cannot be calculated."""

@dataclass(frozen=True)
class SimilarityResult:
    query: str
    document: str
    score: float

# Implements the mathematical formula directly.
def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculate cosine similarity between two equal-length vectors."""

    if len(vector_a) != len(vector_b):
        raise SimilarityError("Embedding vectors must have the same length.")

    if not vector_a or not vector_b:
        raise SimilarityError("Embedding vectors cannot be empty.")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = sum(value * value for value in vector_a) ** 0.5
    magnitude_b = sum(value * value for value in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

# Generates both vectors and returns a typed result.
def compare_similarity(
        provider: OllamaEmbeddingProvider,
        query: str,
        document: str,
) -> SimilarityResult:
    """Generate embeddings and compare semantic similarity."""

    try:
        query_vector = provider.create_embedding(query)
        document_vector = provider.create_embedding(document)
    except EmbeddingError as exc:
        raise  SimilarityError(str(exc)) from exc

    score = cosine_similarity(query_vector, document_vector)

    return SimilarityResult(
        query=query,
        document=document,
        score=round(score, 6),
    )