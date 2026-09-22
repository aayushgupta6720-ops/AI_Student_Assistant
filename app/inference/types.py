"""Provider-neutral message and event types.

Every layer above inference speaks in these types. Nothing outside
app/inference may import a vendor SDK - that boundary is what lets you swap
Gemini for Claude (or a local model) by changing one file."""

from dataclasses import dataclass, field
from typing import Literal, Union


# ---- message parts (what goes *into* the model) ----------------------------


@dataclass
class TextPart:
    text: str
    # Opaque, provider-owned data (e.g. Gemini thought signatures). Layers
    # above inference carry it through untouched and never interpret it.
    provider_state: dict = field(default_factory=dict)


@dataclass
class ToolCallPart:
    """The model asked us to run a tool."""

    id: str
    name: str
    args: dict
    provider_state: dict = field(default_factory=dict)


@dataclass
class ToolResultPart:
    """What we got back from running that tool, fed back to the model."""

    call_id: str
    name: str
    result: dict
    is_error: bool = False


Part = Union[TextPart, ToolCallPart, ToolResultPart]


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    parts: list[Part] = field(default_factory=list)

    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))


@dataclass
class ToolSpec:
    """Provider-neutral tool declaration (JSON Schema for the input)."""

    name: str
    description: str
    input_schema: dict


# ---- stream events (what comes *out* of the model) --------------------------


@dataclass
class TextDelta:
    text: str
    provider_state: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    provider_state: dict = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class StreamEnd:
    finish_reason: str


StreamEvent = Union[TextDelta, ToolCall, Usage, StreamEnd]
