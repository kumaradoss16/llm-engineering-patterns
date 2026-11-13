"""Command-line interface for bounded tool calling."""

import argparse
import json
import sys

from app.patterns.tool_calling import ToolCallError, run_tool_call
from app.providers.ollama import OllamaProvider

def main() -> int:
    parser = argparse.ArgumentParser(prog="tools")
    parser.add_argument("request", help="User request for the technical assistant.")
    args = parser.parse_args()

    provider = OllamaProvider()

    try:
        answer, selection, final, tool_request, tool_result = run_tool_call(
            provider, args.request
        )
    except ToolCallError as exc:
        print(f"Tool call failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(
        {
            "answer": answer,
            "tool_request": tool_request.model_dump(),
            "tool_result": tool_result,
            "selection_latency_ms": round(selection.latency_ms, 2),
            "final_latency_ms": round(final.latency_ms, 2),
            "total_latency_ms": round(
                selection.latency_ms + final.latency_ms, 2
            ),
            "selection_input_tokens": selection.input_tokens,
            "selection_output_tokens": selection.output_tokens,
            "final_input_tokens": final.input_tokens,
            "final_output_tokens": final.output_tokens,
        },
        indent=2,
    ))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

