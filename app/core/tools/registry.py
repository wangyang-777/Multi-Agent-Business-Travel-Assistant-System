"""Tool registry with registration, lookup, package discovery, and MCP catalog."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol, TypeVar

T = TypeVar("T", bound=Callable[..., Awaitable[Any]])


class MCPToolDescriptor(Protocol):
    name: str
    description: str


@dataclass
class RegisteredTool:
    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    json_schema: dict[str, Any] = field(default_factory=dict)
    mcp_compatible: bool = True


class ToolRegistry:
    """Registers agent tools and exposes an MCP-compatible tool catalog."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        handler: T,
        *,
        description: str = "",
        json_schema: dict[str, Any] | None = None,
        mcp_compatible: bool = True,
    ) -> T:
        desc = description or (inspect.getdoc(handler) or "")
        self._tools[name] = RegisteredTool(
            name=name,
            description=desc.strip(),
            handler=handler,
            json_schema=json_schema or {},
            mcp_compatible=mcp_compatible,
        )
        return handler

    def has(self, name: str) -> bool:
        return name in self._tools

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def list_tools(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def mcp_tool_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.json_schema or {"type": "object", "properties": {}},
            }
            for t in self._tools.values()
            if t.mcp_compatible
        ]

    def discover_entry_points(self, group: str = "商旅_agent.tools") -> int:
        eps = entry_points()
        selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
        count = 0
        for ep in selected:
            mod = importlib.import_module(ep.module)
            register_fn = getattr(mod, ep.attr, None)
            if callable(register_fn):
                register_fn(self)
                count += 1
        return count

    def discover_package(self, package: str) -> int:
        """Imports submodules that define `register_tools(registry: ToolRegistry) -> None`."""
        pkg = importlib.import_module(package)
        path = getattr(pkg, "__path__", None)
        if path is None:
            return 0
        prefix = pkg.__name__ + "."
        count = 0
        for _finder, modname, _ispkg in pkgutil.walk_packages(path, prefix):
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            fn = getattr(mod, "register_tools", None)
            if callable(fn):
                fn(self)
                count += 1
        return count


async def invoke(
    registry: ToolRegistry,
    name: str,
    arguments: Mapping[str, Any] | None = None,
) -> Any:
    reg = registry.get(name)
    args = dict(arguments or {})
    result = reg.handler(**args)
    if inspect.isawaitable(result):
        return await result
    return result
