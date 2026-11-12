"""Common interface for model providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ModelResponse:
    text: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    error: str | None = None


class BaseModelProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
        """Send a prompt and return a normalized model response."""