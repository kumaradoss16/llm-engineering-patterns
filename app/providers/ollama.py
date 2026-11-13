"""Ollama provider implementation."""

import os
import time
from typing import Any

try:
    import httpx2 as httpx
except ImportError:
    import httpx

from .base import BaseModelProvider, ModelResponse

class OllamaProvider(BaseModelProvider):
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout_seconds: float = 300.0) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        self.timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT", str(timeout_seconds)))


    def generate(
            self,
            prompt: str,
            temperature: float = 0.0,
    ) -> ModelResponse:
        started = time.perf_counter()

        try:
            # Send the HTTP POST request
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                    "think": False,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

        except httpx.TimeoutException:
            return ModelResponse("", 0.0, None, None, "timeout")
        except httpx.ConnectError:
            return ModelResponse("", 0.0, None, None, "connection_error")
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            error = "rate_limited" if status == 429 else f"http_{status}"
            return ModelResponse("", 0.0, None, None, error)

        latency_ms = (time.perf_counter() - started) * 1000
        payload: dict[str, Any] = response.json()

        return ModelResponse(
            text=payload.get("response", "").strip(),
            latency_ms=latency_ms,
            input_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
        )
