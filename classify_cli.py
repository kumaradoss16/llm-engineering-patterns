"""Command-line interface for ticket and error classification."""

import argparse   # Reads and validates command-line arguments
import json
import sys

from app.patterns.classification import (
    ClassificationError,
    classify_error,
    classify_ticket
)
from app.providers.ollama import OllamaProvider

def main() -> int:
    parser = argparse.ArgumentParser(prog="classify")
    parser.add_argument(
        "kind",
        choices=["ticket", "error"],
        help="Type of text to classify.",
    )
    parser.add_argument("text", help="Text to classify.")
    args = parser.parse_args()   # Reads the arguments supplied by the user.

    provider = OllamaProvider()   # Isolates all Ollama-specific HTTP details.

    try:
        if args.kind == "ticket":
            result, response = classify_ticket(provider, args.text)
            output = {
                "label": result.label,
                "reasoning": result.reasoning,
                "latency_ms": round(response.latency_ms, 2),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        else:
            result, response = classify_error(provider, args.text)
            output = {
                "category": result.category,
                "reasoning": result.reasoning,
                "latency_ms": round(response.latency_ms, 2),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }

        print(json.dumps(output, indent=2))
        return 0
    except ClassificationError as exc:
        print(f"Classification failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
