"""Deterministic IPv4 address extraction using a regular expression."""

import ipaddress
import re


IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)

def extract_ipv4_addresses(text: str) -> list[str]:
    """Extract syntactically valid IPv4 addresses from text."""

    if not text or not text.strip():
        return []

    candidates = IPV4_PATTERN.findall(text)
    valid_addresses: list[str] = []

    for candidate in candidates:
        try:
            ipaddress.IPv4Address(candidate)
        except ValueError:
            continue

        if candidate not in valid_addresses:
            valid_addresses.append(candidate)

    return valid_addresses


def extract_first_ipv4(text: str) -> str | None:
    addresses = extract_ipv4_addresses(text)
    return addresses[0] if addresses else None

