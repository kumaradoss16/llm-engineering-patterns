"""Command-line interface for semantic similarity comparison."""

import argparse
import json
import sys

from app.patterns.similarity import SimilarityError, compare_similarity
from app.providers.embeddings import OllamaEmbeddingProvider


def main() -> int:
    parser = argparse.ArgumentParser(prog="similarity")
    parser.add_argument("query", help="Query text.")
    parser.add_argument("document", help="Document text to remove.")
    args = parser.parse_args()

    provider = OllamaEmbeddingProvider()

    try:
        result = compare_similarity(provider, args.query, args.document)
    except SimilarityError as exc:
        print(f"Similarity calculation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(
        {
            "query": result.query,
            "document": result.document,
            "cosine_similarity": result.score,
        },
        indent=2
    ))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())