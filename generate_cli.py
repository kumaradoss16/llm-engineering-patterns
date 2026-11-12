"""Command-line interface for technical generation."""

import argparse
import json
import sys
from multiprocessing import context
from pathlib import Path

from app import providers
from app.patterns.generation import (
    GenerationError,
    generate_explanation,
    generate_report,
)

from app.providers.ollama import OllamaProvider

def main() -> int:
    parser = argparse.ArgumentParser(prog="generate")
    subparsers = parser.add_subparsers(dest="command", required=True)   # allows one CLI application to support multiple commands.
    explanation = subparsers.add_parser("explanation")
    explanation.add_argument("question")
    explanation.add_argument("--context-file")
    explanation.add_argument("--context")

    report = subparsers.add_parser("report")
    report.add_argument(
        "findings",
        nargs="+",
        help="One or more findins to include in the report."
    )

    args = parser.parse_args()   # Processes the arguments entered by the user.
    provider = OllamaProvider()

    try:
        if args.command == "explanation":
            context = args.context

            if args.context_file:
                context_path = Path(args.context_file)

                if not context_path.exists():
                    raise GenerationError(
                        f"Context file not found: {context_path}"
                    )

                context = context_path.read_text(encoding="utf-8")

            result, response = generate_explanation(
                provider, args.question, context
            )

            output = {
                "title": result.title,
                "explanation": result.explanation,
                "diagnostic_steps": result.diagnostic_steps,
                "cautions":  result.cautions,
                "latency_ms": round(response.latency_ms, 2),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }

        else:
            result, response = generate_report(provider, args.findings)

            output = {
                "title": result.title,
                "summary": result.summary,
                "findings": result.findings,
                "next_steps": result.next_steps,
                "latency_ms": round(response.latency_ms, 2),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }

        print(json.dumps(output, indent=2))
        return 0

    except GenerationError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())




