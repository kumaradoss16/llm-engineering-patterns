"""Command-line interface for technical summarization."""

import argparse
import json
import sys
from pathlib import Path

from app.patterns.summarization import SummarizationError, summarize_text
from app.providers.ollama import OllamaProvider

def main() -> int:
    parser = argparse.ArgumentParser(prog="summarize")
    parser.add_argument("input_file", help="Path to a UTF-8 text file.")
    parser.add_argument(
        "--style",
        default="three action-oriented bullet points",
        help="Requested summary style."
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        print(f"Input file must be UTF-8 encoded: {exc}", file=sys.stderr)
        return 1

    provider = OllamaProvider()

    try:
        result, response = summarize_text(provider, text, args.style)
    except SummarizationError as exc:
        print(f"Summarization failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(
        {
            "summary": result.summary,
            "key_points": result.key_points,
            "latency_ms": round(response.latency_ms, 2),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
        indent=2
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
