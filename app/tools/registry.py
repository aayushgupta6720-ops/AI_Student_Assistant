"""Tool registry: the tools layer's public surface.

A Tool is a name + description + JSON schema + async handler. The registry
turns tools into provider-neutral ToolSpecs (for the inference layer) and
executes calls by name, catching exceptions so a broken tool becomes an
error result the model can read rather than a crashed request."""

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.inference.types import ToolSpec
from app.observability import time_step

Handler = Callable[..., Awaitable[dict] | dict]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Handler

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)


@dataclass
class ToolOutcome:
    name: str
    result: dict
    is_error: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    async def execute(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome(name, {"error": f"unknown tool: {name}"}, is_error=True)
        with time_step("tools", name, args=args):
            try:
                result = tool.handler(**args)
                if inspect.isawaitable(result):
                    result = await result
                return ToolOutcome(name, result)
            except Exception as exc:  # noqa: BLE001 - surface any failure to the model
                return ToolOutcome(name, {"error": f"{type(exc).__name__}: {exc}"}, is_error=True)
