"""Deterministic extraction helpers that do not require an LLM."""

import ipaddress
import re

IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)

PORT_PATTERN = re.compile(r"\b(?:port\s+|:)(\d{1,5})\b", re.IGNORECASE)


# Extracts the first IPv4-pattern match from a string
def extract_ipv4(text: str) -> str | None:
    match = IPV4_PATTERN.search(text)
    return match.group(0) if match else None


def extract_port(text: str) -> int | None:
    match = PORT_PATTERN.search(text)
    if not match:
        return None

    port = int(match.group(1))
    return port if 1 <= port <= 65535 else None


def is_valid_ipv4(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
        return True
    except ValueError:
        return False