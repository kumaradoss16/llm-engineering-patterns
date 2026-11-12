"""Embedding provider for local Ollama models"""

import os
import httpx2


class EmbeddingError(Exception):
    """Raised when an embedding cannot be generated."""


class OllamaEmbeddingProvider:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:

        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"
        )
        self.timeout_seconds = timeout_seconds

    # Validates input and checks that the response container a list.
    def create_embedding(self, text: str) -> list[float]:
        cleaned = text.strip()

        if not cleaned:
            raise EmbeddingError("Text cannot be empty.")

        if len(cleaned) > 10_000:
            raise EmbeddingError("Text exceeds the 10,000 character limit.")

        try:
            response = httpx2.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": cleaned,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

        except httpx2.TimeoutException as exc:
            raise EmbeddingError("Embedding request time out.") from exc
        except httpx2.ConnectError as exc:
            raise EmbeddingError(
                "Cannot connect to local Ollama embedding service."
            ) from exc
        except httpx2.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc

        payload = response.json()
        embedding = payload.get("embedding")

        if not embedding or not isinstance(embedding, list):
            raise EmbeddingError("Ollama returned an invalid embedding response.")

        return embedding