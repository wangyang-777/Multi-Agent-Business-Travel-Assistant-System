from __future__ import annotations

import httpx
import pytest

from app.utils.openai_client import create_openai_client


@pytest.mark.asyncio
async def test_openai_client_accepts_host_with_socks5h_all_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")
    monkeypatch.setenv("ALL_PROXY", "socks5h://127.0.0.1:10800")
    monkeypatch.setenv("NO_PROXY", "")

    client = create_openai_client(api_key="test", base_url="https://example.com/v1")
    try:
        assert isinstance(client._client, httpx.AsyncClient)
    finally:
        await client.close()
