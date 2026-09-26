"""A scripted LLMProvider for tests: no network, deterministic."""

from typing import AsyncIterator

from app.inference.types import Message, StreamEnd, StreamEvent, TextDelta, ToolCall, ToolSpec, Usage


class FakeProvider:
    def __init__(self, turns: list[list[StreamEvent]], dim: int = 4) -> None:
        self.turns = list(turns)
        self.dim = dim
        self.seen: list[list[Message]] = []
        self.systems: list[str] = []
        self.tools_seen: list[list[str]] = []  # the tool names each call was offered

    async def stream_generate(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[StreamEvent]:
        self.seen.append(list(messages))
        self.systems.append(system)
        self.tools_seen.append([t.name for t in tools])
        for event in self.turns.pop(0):
            yield event
        yield Usage(input_tokens=10, output_tokens=5)
        yield StreamEnd(finish_reason="stop")

    async def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        # Deterministic "embedding": character histogram over 4 buckets.
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for ch in t.lower():
                v[ord(ch) % self.dim] += 1
            out.append(v)
        return out


def text_turn(text: str) -> list[StreamEvent]:
    return [TextDelta(text)]


def tool_turn(name: str, args: dict, call_id: str = "c1") -> list[StreamEvent]:
    return [ToolCall(call_id, name, args)]
