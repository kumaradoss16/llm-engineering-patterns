"""Evaluate AI application patterns using a fixed dataset.

This script records task-specific results. It does not fabricate quality
scores: classification and extraction are checked deterministically, while
summarization and generation require human review.
"""

import json
import time
from pathlib import Path
from typing import Any

from app.deterministic.ip_extractor import extract_first_ipv4
from app.patterns.classification import classify_ticket
from app.patterns.extraction import extract_log_information
from app.patterns.rag import InMemoryKnowledgeBase, answer_question
from app.patterns.summarization import summarize_text
from app.providers.embeddings import OllamaEmbeddingProvider
from app.providers.ollama import OllamaProvider


def load_dataset(path: str = "evaluation_dataset.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def evaluate_classification(
        provider: OllamaProvider,
        test_case: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        result, response = classify_ticket(
            provider, test_case["input"]
        )
        correct = result.label == test_case["expected_label"]
        error = None
    except Exception as exc:
        correct = False
        error = str(exc)
        response = None

    latency = (time.perf_counter() - started) * 1000

    return {
        "id": test_case["id"],
        "pattern": test_case["pattern"],
        "correct": correct,
        "latency_ms": round(latency, 2),
        "input_tokens": response.input_tokens if response else None,
        "output_tokens": response.output_tokens if response else None,
        "error": error,
    }


def evaluate_extraction(
        provider: OllamaProvider,
        test_case: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        result, response = extract_log_information(
            provider, test_case["input"]
        )
        extracted = result.model_dump()
        expected = test_case["expected_fields"]

        correct = all(
            extracted.get(field) == value
            for field, value in expected.items()
        )
        error = None
    except Exception as exc:
        correct = False
        extracted = {}
        error = str(exc)
        response = None
    latency = (time.perf_counter() - started) * 1000

    return {
        "id": test_case["id"],
        "pattern": test_case["pattern"],
        "correct": correct,
        "extracted": extracted,
        "latency_ms": round(latency, 2),
        "input_tokens": response.input_tokens if response else None,
        "output_tokens": response.output_tokens if response else None,
        "error": error,
    }

def evaluate_summarization(
        provider: OllamaProvider,
        test_case: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        result, response = summarize_text(
            provider, test_case["input"]
        )
        summary_lower = result.summary.lower()
        covered = sum(
            1
            for keyword in test_case["expected_keywords"]
            if keyword.lower() in summary_lower
        )
        correct = covered == len(test_case["expected_keywords"])
        error = None
    except Exception as exc:
        correct = False
        covered = 0
        error = str(exc)
        response = None

    latency = (time.perf_counter() - started) * 1000

    return {
        "id": test_case["id"],
        "pattern": test_case["pattern"],
        "correct": correct,
        "keyword_coverage": f"{covered}/{len(test_case['expected_keywords'])}",
        "latency_ms": round(latency, 2),
        "input_tokens": response.input_tokens if response else None,
        "output_tokens": response.output_tokens if response else None,
        "error": error,
    }

def evaluate_rag(
        provider: OllamaProvider,
        embedding_provider: OllamaEmbeddingProvider,
        test_case: dict[str, Any],
) -> dict[str, Any]:
    knowledge_base = InMemoryKnowledgeBase(chunks=[])

    for document_path in sorted(
        Path("sample_data/knowledge").glob("*.md")
    ):
        knowledge_base.add_document(
            document_id=document_path.stem,
            title=document_path.stem.replace("-", " ").title(),
            text=document_path.read_text(encoding="utf-8"),
            embedding_provider=embedding_provider,
        )

    started = time.perf_counter()

    try:
        answer, response, retrieved = answer_question(
            generator=provider,
            knowledge_base=knowledge_base,
            embedding_provider=embedding_provider,
            question=test_case["input"],
        )

        if test_case["evaluation"] == "abstention":
            correct = (
                answer.answer.lower().startswith(
                    "the answer is not available"
                )
                and not answer.sources
            )
        else:
            source_ids = [
                source.strip("[]") for source in answer.sources
            ]
            retrieved_ids = [
                chunk.document_id for chunk in retrieved
            ]

            correct = (
                test_case["expected_source"] in retrieved_ids
                and test_case["expected_answer_container"].lower()
                in answer.answer.lower()
            )

        error = None
    except Exception as exc:
        correct = False
        error = str(exc)
        response = None
        retrieved = []

    latency = (time.perf_counter() - started) * 1000

    return {
        "id": test_case["id"],
        "pattern": test_case["pattern"],
        "correct": correct,
        "retrieved_documents": [
            chunk.document_id for chunk in retrieved
        ],
        "latency_ms": round(latency, 2),
        "input_tokens": response.input_tokens if response else None,
        "output_tokens": response.output_tokens if response else None,
        "error": error,
    }

def evaluate_deterministic(
        test_case: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    extracted = extract_first_ipv4(test_case["input"])
    latency = (time.perf_counter() - started) * 1000

    return {
        "id": test_case["id"],
        "pattern": test_case["pattern"],
        "correct": extracted == test_case["expected_value"],
        "extracted": extracted,
        "latency_ms": round(latency, 2),
        "input_tokens": None,
        "output_tokens": None,
        "error": None,
    }

def main() -> int:
    dataset = load_dataset()
    provider = OllamaProvider()
    embedding_provider = OllamaEmbeddingProvider()

    results: list[dict[str, Any]] = []

    for test_case in dataset["test_cases"]:
        pattern = test_case["pattern"]

        if pattern == "classification":
            results.append(evaluate_classification(provider, test_case))
        elif pattern == "extraction":
            results.append(evaluate_extraction(provider, test_case))
        elif pattern == "summarization":
            results.append(evaluate_summarization(provider, test_case))
        elif pattern == "rag":
            results.append(evaluate_rag(provider, embedding_provider, test_case))
        elif pattern == "deterministic":
            results.append(evaluate_deterministic(test_case))
        elif pattern == "tool_calling":
            results.append(
                {
                    "id": test_case["id"],
                    "pattern": pattern,
                    "correct": None,
                    "error": "Requires manual or mocked tool call evaluation.",
                }
            )

    output_path = Path("results/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"Results saved to {output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())






