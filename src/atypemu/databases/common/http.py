"""HTTP helpers shared by database-source crawlers."""

from __future__ import annotations

from typing import Any

import requests


DEFAULT_USER_AGENT = "AtypEmu/0.1"


def fetch_url(
    url: str,
    *,
    timeout_seconds: int,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """Fetch one URL with a consistent AtypEmu user-agent header."""
    request_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers is not None:
        request_headers.update(headers)
    return requests.get(url, timeout=timeout_seconds, headers=request_headers)


def response_metadata(response: requests.Response) -> dict[str, Any]:
    """Extract a compact metadata payload from one response object."""
    return {
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "content_length": response.headers.get("content-length"),
        "url": response.url,
    }
