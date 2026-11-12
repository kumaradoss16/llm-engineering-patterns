"""Latency measurement utilities."""

import time
from contextlib import contextmanager
from collections.abc import Iterator

@contextmanager
def measure_latency_ms() -> Iterator[dict[str, float]]:
    started = time.perf_counter()
    metrics: dict[str, float] = {"latency_ms": 0.0}

    try:
        yield metrics
    finally:
        metrics["latency_ms"] = round(
            (time.perf_counter() - started) * 1000, 2
        )