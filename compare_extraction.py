"""Compare deterministic regex extraction with LLM-based extraction"""

import argparse
import json
import time
from dataclasses import asdict, dataclass

from app.deterministic.ip_extractor import extract_first_ipv4
from app.patterns.llm_extractor import (
    LLMExtractionError,
    IPAddressExtraction,
    extract_ip_with_llm,
)
from app.providers.ollama import OllamaProvider


@dataclass
class ComparisonResult:
    approach: str
    input_text: str
    extracted_value: str | None
    correct: bool
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    error: str | None


EXPECTED_IP = "192.168.1.25"
TEST_TEXT = (
    "Connection failed to 192.168.1.25:5432 after 3 retires."
    "Please investigate the database connectivity issue."
)

ILLUSTRATIVE_PRICING = {
    "input_per_million": 0.15,
    "output_per_million": 0.69,
}

def estimate_llm_cost(
        input_tokens: int | None,
        output_tokens: int | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None

    input_cost = (
        input_tokens * ILLUSTRATIVE_PRICING["input_per_million"] / 1_000_000
    )

    output_cost = (
            output_tokens * ILLUSTRATIVE_PRICING["output_per_million"] / 1_000_000
    )

    return round(input_cost + output_cost, 8)


def run_deterministic() -> ComparisonResult:
    started = time.perf_counter()
    extracted = extract_first_ipv4(TEST_TEXT)
    latency_ms = (time.perf_counter() - started) * 1000

    return ComparisonResult(
        approach="deterministic-regex",
        input_text=TEST_TEXT,
        extracted_value=extracted,
        correct=extracted == EXPECTED_IP,
        latency_ms=round(latency_ms, 4),
        input_tokens=None,
        output_tokens=None,
        estimated_cost_usd=0.0,
        error=None,
    )

def run_llm() -> ComparisonResult:
    provider = OllamaProvider()
    started = time.perf_counter()

    try:
        result, response = extract_ip_with_llm(provider, TEST_TEXT)
    except LLMExtractionError as exc:
        return ComparisonResult(
            approach="llm-extraction",
            input_text=TEST_TEXT,
            extracted_value=None,
            correct=False,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
            error=str(exc),
        )

    latency_ms = (time.perf_counter() - started) * 1000

    return ComparisonResult(
        approach="llm-extraction",
        input_text=TEST_TEXT,
        extracted_value=result.ip_address,
        correct=result.ip_address == EXPECTED_IP,
        latency_ms=round(latency_ms, 2),
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        estimated_cost_usd=estimate_llm_cost(
            response.input_tokens,
            response.output_tokens,
        ),
        error=None,
    )

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="compare-extraction",
        description="Compare deterministic and LLM-based IP extraction.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of LLM repetitions."
    )
    args = parser.parse_args()

    deterministic_result = run_deterministic()
    print(json.dumps(asdict(deterministic_result), indent=2))

    for repetition in range(1, args.repeat + 1):
        llm_result = run_llm()
        llm_result.approach = f"llm-extraction-run-{repetition}"
        print(json.dumps(asdict(llm_result), indent=2))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())