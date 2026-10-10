"""Create OpenAI-compatible clients in hosts that use SOCKS proxy URLs."""

from __future__ import annotations

import os
from urllib.parse import urlparse
from urllib.request import proxy_bypass

import httpx
from openai import AsyncOpenAI


def create_async_http_client(*, url: str, timeout: float) -> httpx.AsyncClient:
    """Create an HTTP client, normalizing SOCKS proxy URLs used by this host."""
    parsed = urlparse(url)
    proxy = (
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if parsed.scheme == "https"
        else os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    ) or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
    has_unsupported_proxy = any(
        value.lower().startswith("socks5h://")
        for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
        if (value := os.environ.get(name))
    )
    options: dict[str, object] = {}
    if has_unsupported_proxy:
        normalized_proxy = proxy if not proxy_bypass(parsed.hostname or "") else None
        if normalized_proxy and normalized_proxy.lower().startswith("socks5h://"):
            normalized_proxy = "socks5://" + normalized_proxy[len("socks5h://"):]
        options = {"proxy": normalized_proxy, "trust_env": False}
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=5.0),
        follow_redirects=True,
        **options,
    )


def create_openai_client(*, api_key: str, base_url: str) -> AsyncOpenAI:
    options: dict[str, object] = {}
    parsed = urlparse(base_url)
    proxy = (
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if parsed.scheme == "https"
        else os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    ) or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
    has_unsupported_proxy = any(
        value.lower().startswith("socks5h://")
        for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
        if (value := os.environ.get(name))
    )
    if has_unsupported_proxy:
        # HTTPX inspects every proxy env var, including ALL_PROXY, during client setup.
        normalized_proxy = proxy if not proxy_bypass(parsed.hostname or "") else None
        if normalized_proxy and normalized_proxy.lower().startswith("socks5h://"):
            normalized_proxy = "socks5://" + normalized_proxy[len("socks5h://"):]
        options["http_client"] = httpx.AsyncClient(
            proxy=normalized_proxy,
            trust_env=False,
            timeout=httpx.Timeout(600.0, connect=5.0),
            follow_redirects=True,
        )
    return AsyncOpenAI(api_key=api_key, base_url=base_url, **options)
