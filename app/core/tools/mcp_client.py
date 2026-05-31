"""JSON-RPC client for MCP (Model Context Protocol) over HTTP."""

from __future__ import annotations

from typing import Any

import httpx


class MCPRpcError(RuntimeError):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(f"MCP RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPHttpClient:
    """MCP client using JSON-RPC 2.0 POST; compatible with HTTP-based MCP servers."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._rpc_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers or {})
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        response = await self._client.post(self._rpc_url, json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            return body
        if "error" in body:
            err = body["error"]
            if isinstance(err, dict):
                raise MCPRpcError(
                    code=int(err.get("code", -1)),
                    message=str(err.get("message", "unknown")),
                    data=err.get("data"),
                )
            raise RuntimeError(f"MCP error: {err}")
        return body.get("result")

    async def initialize(self, client_name: str = "商旅-agent-guide") -> dict[str, Any]:
        return await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "0.1.0"},
            },
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._rpc("tools/list")
        if isinstance(result, dict) and "tools" in result:
            return list(result["tools"])
        if isinstance(result, list):
            return result
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})

    async def list_resources(self) -> list[dict[str, Any]]:
        result = await self._rpc("resources/list")
        if isinstance(result, dict) and "resources" in result:
            return list(result["resources"])
        if isinstance(result, list):
            return result
        return []

    async def read_resource(self, uri: str) -> dict[str, Any]:
        raw = await self._rpc("resources/read", {"uri": uri})
        return raw if isinstance(raw, dict) else {"contents": raw}

    async def ping(self) -> bool:
        try:
            await self._client.get(self._rpc_url.replace("/rpc", "/health"), timeout=5.0)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


MCPClient = MCPHttpClient
