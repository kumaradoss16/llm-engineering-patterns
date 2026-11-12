"""Command-line interface for technical information extraction."""

import argparse
import json
import sys

from app import providers, deterministic
from app.deterministic.extractors import extract_ipv4, extract_port
from app.patterns.extraction import (
    ExtractionError,
    extract_log_information,
    extract_software_information
)
from app.providers.ollama import OllamaProvider

def main() -> int:
    parser = argparse.ArgumentParser(prog="extract")
    parser.add_argument(
        "kind",
        choices=("log", "software"),
        help="Type of extraction to perform."
    )
    parser.add_argument("text", help="Text to process.")
    args = parser.parse_args()

    provider = OllamaProvider()

    try:
        if args.kind == "log":
            result, response = extract_log_information(provider, args.text)
            deterministic_ip = extract_ipv4(args.text)
            deterministic_port = extract_port(args.text)

            output = {
                "model_extraction": result.model_dump(),   # converts the Pydantic model into a standard Python dictionary.
                "deterministic_ip": deterministic_ip,
                "deterministic_port": deterministic_port,
                "latency_md": round(response.latency_ms, 2),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        else:
            result, response = extract_software_information(provider, args.text)
            output = {
                "extraction": result.model_dump(),
                "latency_ms": round(response.latency_ms, 2),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        print(json.dumps(output, indent=2))   # converts the Python dictionary into formatted JSON
        return 0
    except ExtractionError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())




