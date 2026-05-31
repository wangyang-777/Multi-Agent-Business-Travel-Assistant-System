from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request

from app.agent.orchestrator import _travel_tools

router = APIRouter(tags=["mcp"])


_TOOLS = [
    {
        "name": tool["function"]["name"],
        "description": tool["function"].get("description", ""),
        "inputSchema": tool["function"].get("parameters", {"type": "object"}),
    }
    for tool in _travel_tools()
]


@router.post("/mcp/rpc")
async def mcp_rpc(body: dict[str, Any], request: Request) -> dict[str, Any]:
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params") if isinstance(body.get("params"), dict) else {}
    try:
        result = await _dispatch(method, params, request)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(exc)},
        }


async def _dispatch(method: str | None, params: dict[str, Any], request: Request) -> Any:
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "travel-agent-guide", "version": "0.1.0"},
            "capabilities": {"tools": {}, "resources": {}},
        }
    if method == "tools/list":
        return {"tools": _TOOLS}
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        orchestrator = request.app.state.orchestrator
        output = await orchestrator._execute_tool(name, json.dumps(arguments))
        return {"content": [{"type": "text", "text": output}]}
    if method == "resources/list":
        return {
            "resources": [
                {
                    "uri": "travel-agent://health",
                    "name": "Travel Agent Health",
                    "description": "当前差旅 Agent 服务状态摘要。",
                    "mimeType": "application/json",
                }
            ]
        }
    if method == "resources/read":
        uri = str(params.get("uri") or "")
        if uri != "travel-agent://health":
            raise ValueError(f"unknown resource: {uri}")
        milvus = getattr(request.app.state, "milvus", None)
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(
                        {
                            "milvus": bool(milvus and milvus.connected),
                            "orchestrator": "langgraph",
                        },
                        ensure_ascii=False,
                    ),
                }
            ]
        }
    raise ValueError(f"unsupported MCP method: {method}")
