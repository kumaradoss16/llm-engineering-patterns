"""Technical AI Application Patterns Lab CLI."""

import argparse
import json
import sys
from pathlib import Path
from app.config import load_settings
from app.deterministic.ip_extractor import extract_first_ipv4

from app.patterns.classification import (
    ClassificationError,
    classify_error,
    classify_ticket,
)
from app.patterns.extraction import (
    ExtractionError,
    extract_log_information,
    extract_software_information,
)
from app.patterns.generation import (
    GenerationError,
    generate_explanation,
)
from app.patterns.rag import RagError, InMemoryKnowledgeBase, answer_question
from app.patterns.similarity import SimilarityError, compare_similarity
from app.patterns.summarization import SummarizationError, summarize_text
from app.patterns.tool_calling import ToolCallError, run_tool_call
from app.providers.embeddings import EmbeddingError, OllamaEmbeddingProvider
from app.providers.ollama import OllamaProvider
from app.utilities.cost import ModelPricing, estimate_cost
from app.utilities.logging import configure_logging

logger = configure_logging()

ILLUSTRATIVE_PRICING = ModelPricing(
    input_per_million=__import__("decimal").Decimal("0.15"),
    output_per_million=__import__("decimal").Decimal("0.60"),
)


# Read a text file and returns its contents.
def read_text_file(path: str) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    return file_path.read_text(encoding="utf-8")

# Creates CLI Interface
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="technical-ai-patterns-lab",
        description=(
            "Demonstrate classification, extraction, summarization, "
            "generation, embeddings, RAG, tool calling, and deterministic "
            "engineering patterns."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("input_file")

    classify = subparsers.add_parser("classify")
    classify.add_argument("kind", choices=["ticket", "error"])
    classify.add_argument("text")

    extract = subparsers.add_parser("extract")
    extract.add_argument("kind", choices=["log", "software"])
    extract.add_argument("text")

    generate = subparsers.add_parser("generate")
    generate.add_argument("question")

    similarity = subparsers.add_parser("similarity")
    similarity.add_argument("query")
    similarity.add_argument("document")

    rag = subparsers.add_parser("rag")
    rag.add_argument("question")
    rag.add_argument("--top-k", type=int, default=2)
    rag.add_argument("--minimum-score", type=float, default=0.0)

    tools = subparsers.add_parser("tools")
    tools.add_argument("request")

    compare = subparsers.add_parser("compare")
    compare.add_argument("text")

    return parser

# Adds usage information to an existing output dictionary.
def add_usage(
        output: dict[str, object],
        latency_ms: float,
        input_tokens: int | None,
        output_tokens: int | None,
) -> None:
    cost = estimate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        pricing=ILLUSTRATIVE_PRICING,
    )

    output["usage"] = {
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_uds": (
            str(cost.total_cost) if cost else None
        ),
    }

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()

    provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    embedding_provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )

    try:
        if args.command == "summarize":
            text = read_text_file(args.input_file)
            result, response = summarize_text(provider, text)
            output: dict[str, object] = {
                "summary": result.summary,
                "key_points": result.key_points,
            }
            add_usage(
                output,
                response.latency_ms,
                response.input_tokens,
                response.output_tokens
            )

        elif args.command == "classify":
            if args.kind == "ticket":
                result, response = classify_ticket(provider, args.text)
                output = {
                    "label": result.label,
                    "reasoning": result.reasoning,
                }
            else:
                result, response = classify_error(provider, args.text)
                output = {
                    "category": result.category,
                    "reasoning": result.reasoning,
                }

            add_usage(
                output,
                response.latency_ms,
                response.input_tokens,
                response.output_tokens
            )

        elif args.command == "extract":
            if args.kind == "log":
                result, response = extract_log_information(
                    provider, args.text
                )
                deterministic_ip = extract_first_ipv4(args.text)
                output = {
                    "model_extraction": result.model_dump(),
                    "deterministic_ip": deterministic_ip,
                }
            else:
                result, response = extract_software_information(
                    provider, args.text
                )
                output = {
                    "extraction": result.model_dump(),
                }

            add_usage(
                output,
                response.latency_ms,
                response.input_tokens,
                response.output_tokens
            )

        elif args.command == "generate":
            result, response = generate_explanation(
                provider, args.question
            )
            output = {
                "title": result.title,
                "explanation": result.explanation,
                "diagnostic_steps": result.diagnostic_steps,
                "cautions": result.cautions,
            }

            add_usage(
                output,
                response.latency_ms,
                response.input_tokens,
                response.output_tokens
            )
        elif args.command == "generate":
            result = compare_similarity(
                embedding_provider, args.query, args.document
            )
            output = {
                "query": result.query,
                "document": result.document,
                "cosine_similarity": result.score,
            }
        elif args.command == "rag":
            knowledge_base = InMemoryKnowledgeBase(chunks=[])

            for document_path in sorted(
                Path("sample_date/knowledge").glob("*.md")
            ):
                knowledge_base.add_document(
                    document_id=document_path.stem,
                    title=document_path.stem.replace("-", " ").title(),
                    text=document_path.read_text(encoding="utf-8"),
                    embedding_provider=embedding_provider,
                )

            answer, response, retrieved = answer_question(
                generator=provider,
                knowledge_base=knowledge_base,
                embedding_provider=embedding_provider,
                question=args.question,
                top_k=args.top_k,
                minimum_score=args.minimum_score,
            )

            output = {
                "answer": answer.answer,
                "sources": answer.sources,
                "confidence": answer.confidence,
                "retrieved_chunks": [
                    {
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.chunk_id,
                        "title": chunk.title,
                        "socre": chunk.score,
                    }
                    for chunk in retrieved
                ],
            }
            add_usage(
                output,
                response.latency_ms,
                response.input_tokens,
                response.output_tokens
            )

        elif args.command == "tools":
            answer, selection, final, tool_request, tools_result = (
                run_tool_call(provider, args.request)
            )

            output = {
                "answer": answer,
                "tool_request": tool_request,
                "total_request": round(
                    selection.latency_ms + final.latency_ms,
                    2,
                ),
                "selection_input_tokens": selection.input_tokens,
                "selection_output_tokens": selection.output_tokens,
                "final_intput_tokens": final.input_tokens,
                "final_output_tokens": final.output_tokens,
            }

        elif args.command == "compare":
            deterministic_ip = extract_first_ipv4(args.text)

            try:
                from app.patterns.llm_extraction import extract_ip_with_llm

                llm_result, response = extract_ip_with_llm(
                    provider, args.text
                )
                llm_ip = llm_result.ip_address
                llm_latency = response.latency_ms
                llm_input_tokens = response.input_tokens
                llm_output_tokens = response.output_tokens
                llm_error = None
            except Exception as exc:
                llm_ip = None
                llm_latency = 0.0
                llm_input_tokens = None
                llm_output_tokens = None
                llm_error = str(exc)
            output = {
                "deterministic_ip": deterministic_ip,
                "llm_ip": llm_ip,
                "llm_latency_ms": round(llm_latency, 2),
                "llm_input_tokens": llm_input_tokens,
                "llm_output_tokens": llm_output_tokens,
                "llm_error": llm_error,
            }

        else:
            raise ValueError(f"Unsupported command: {args.command}")

        print(json.dumps(output, indent=2))
        return 0

    except (
        ClassificationError,
        ExtractionError,
        SummarizationError,
        GenerationError,
        SimilarityError,
        RagError,
        EmbeddingError,
        ToolCallError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        logger.error("Command failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())




















