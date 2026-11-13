"""Command-line interface for the basic RAG demonstration."""

import argparse
import json
import sys
from pathlib import Path

from app.patterns.rag import (
    InMemoryKnowledgeBase,
    RagError,
    answer_question,
)

from app.providers.embeddings import OllamaEmbeddingProvider
from app.providers.ollama import OllamaProvider
from app.providers.embeddings import EmbeddingError


def load_knowledge_base(
        embedding_provider: OllamaEmbeddingProvider,
) -> InMemoryKnowledgeBase:
    knowledge_base = InMemoryKnowledgeBase(chunks=[])   # Initialized the knowledge base
    knowledge_dir = Path("sample_data/knowledge")

    if not knowledge_dir.exists():
        raise RagError(f"Knowledge directory not found: {knowledge_dir}")

    markdown_files = sorted(knowledge_dir.glob("*.md"))   # Find the Markdown file

    if not markdown_files:
        raise RagError("No Markdown knowledge documents were found.")

    for document_path in markdown_files:
        text = document_path.read_text(encoding="utf-8")
        knowledge_base.add_document(
            document_id=document_path.stem,   # Return the filename without its extension
            title=document_path.stem.replace("-", " ").title(),
            text=text,
            embedding_provider=embedding_provider,
        )

    return knowledge_base

def main() -> int:
    parser = argparse.ArgumentParser(prog="rag")
    parser.add_argument("question", help='Question to answer from the knowledge base.')
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--minimum-score", type=float, default=0.0)
    args = parser.parse_args()   # Reads the supplied command-line arguments and stores them in a namespace object.

    embedding_provider = OllamaEmbeddingProvider()   # Converts text into vectors
    generator = OllamaProvider()   # Generates a human-readable answer

    try:
        knowledge_base = load_knowledge_base(embedding_provider)
        answer, response, retrieved = answer_question(
            generator=generator,
            knowledge_base=knowledge_base,
            embedding_provider=embedding_provider,
            question=args.question,
            top_k=args.top_k,
            minimum_score=args.minimum_score,
        )
    except (RagError, EmbeddingError) as exc:
        print(f"RAG failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(
        {
            "answer": answer.answer,
            "sources": answer.sources,
            "confidence": answer.confidence,
            "retrieved_chunks": [
                {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "score": chunk.score,
                }
                for chunk in retrieved
            ],
            "latency_ms": round(response.latency_ms, 2),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
        indent=2,
    ))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
