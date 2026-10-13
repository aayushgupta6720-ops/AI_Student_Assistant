"""Tool registry: the tools layer's public surface.

A Tool is a name + description + JSON schema + async handler. The registry
turns tools into provider-neutral ToolSpecs (for the inference layer) and
executes calls by name, catching exceptions so a broken tool becomes an
error result the model can read rather than a crashed request."""

import inspect
import json
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


def _whole(value: Any) -> bool:
    """A whole number, however the model wrote it: 3, 3.0 (Gemini sends
    integers as floats) or "3" (the handlers convert it). Not True, which
    Python counts as 1."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return False
    return not isinstance(value, bool) and isinstance(value, (int, float)) and float(value).is_integer()


_JSON_TYPES: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": _whole,
    "number": lambda v: not isinstance(v, bool) and isinstance(v, (int, float)),
    "boolean": lambda v: isinstance(v, bool),
}


def args_problem(schema: dict, args: Any) -> str | None:
    """Why `args` don't fit a tool's schema, or None if they do: only the
    arguments the schema lists, every required one, each of its type. The
    model's arguments go straight into the handler, so without this a handler
    parameter the schema never offered (fetch_url's max_chars once was) is
    still the model's to set. A schema with no "properties" at all doesn't
    limit the names. Ranges are left to the handlers, which clamp."""
    if not isinstance(args, dict):
        return "expected an object of named arguments"
    properties = schema.get("properties")
    if properties is None:
        properties = {}
    else:
        unknown = sorted(set(args) - set(properties))
        if unknown:
            return f"unknown argument(s) {', '.join(unknown)}; this tool takes {', '.join(properties) or 'none'}"
    missing = [key for key in schema.get("required", []) if key not in args]
    if missing:
        return f"missing required argument(s) {', '.join(missing)}"
    for key, value in args.items():
        expected = properties.get(key, {}).get("type")
        check = _JSON_TYPES.get(expected)
        if check and not check(value):
            return f"{key} must be a{'n' if expected[0] in 'aeiou' else ''} {expected}"
    return None


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
        problem = args_problem(tool.input_schema, args)
        if problem:
            return ToolOutcome(name, {"error": f"invalid arguments: {problem}"}, is_error=True)
        with time_step("tools", name, args=args):
            try:
                result = tool.handler(**args)
                if inspect.isawaitable(result):
                    result = await result
                # The result is kept in the chat's history and sent back to the
                # model as JSON on every later turn, so one that can't be sent
                # (a nan, a complex number) would break the chat for good.
                json.dumps(result, allow_nan=False)
                return ToolOutcome(name, result)
            except Exception as exc:  # noqa: BLE001 - surface any failure to the model
                return ToolOutcome(name, {"error": f"{type(exc).__name__}: {exc}"}, is_error=True)
