"""Minimal semantic search over a small local document collection."""

import json
import sys
from pathlib import Path

from pip._internal.resolution.resolvelib import provider

from app.patterns.similarity import (
    SimilarityError,
    compare_similarity,
)
from app.providers.embeddings import OllamaEmbeddingProvider

DOCUMENTS = [
    {
        "id": "nginx-restart",
        "title": "Restarting nginx",
        "text": "Use systemctl restart nginx to restart the nginx service on Linux.",
    },
    {
        "id": "postgres-connectivity",
        "title": "PostgreSQL connectivity",
        "text": "Check database host, port, credentials, firewall rules, and network policy when an application cannot connect to PostgreSQL.",
    },
    {
        "id": "docker-dns",
        "title": "Docker DNS issues",
        "text": "Container DNS failures can occur when the Docker network, DNS configuration, or upstream resolver is unavailable.",
    },
]

def search(query: str, minimum_score: float = 0.0) -> list[dict[str, object]]:
    provider = OllamaEmbeddingProvider()
    results: list[dict[str, object]] = []

    for document in DOCUMENTS:
        similarity = compare_similarity(
            provider,
            query,
            document["text"],
        )

        if similarity.score >= minimum_score:
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "score": similarity.score,
                    "text": document["text"],
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python semantic_search.py \"search query\"")
        return 1

    query = sys.argv[1]   # Python list containing the command-line arguments passed to the program

    try:
        results = search(query)
    except SimilarityError as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(results, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
