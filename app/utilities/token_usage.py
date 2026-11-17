"""Token usage utilities.

Token counts are provider and model-specific. If a provider does not expose
usage metadata, record None rather than estimating or inventing a value.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None
    output_tokens: int | None

    @property   # allows you to access a method like an ordinary attribute.
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None

        return self.input_tokens + self.output_tokens


def build_token_usage(
        input_tokens: int | None,
        output_tokens: int | None,
) -> TokenUsage:
    if input_tokens is not None and input_tokens < 0:
        raise ValueError("Input token count cannot be negative.")

    if output_tokens is not None and output_tokens < 0:
        raise ValueError("output token count cannot be negative.")

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )