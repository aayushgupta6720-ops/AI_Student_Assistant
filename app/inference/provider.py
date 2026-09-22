from typing import AsyncIterator, Literal, Protocol

from app.inference.types import Message, StreamEvent, ToolSpec

EmbedKind = Literal["document", "query"]


class LLMProvider(Protocol):
    """The inference layer's contract. Two capabilities: generate (streamed,
    with tool calling) and embed. Implementations: GeminiProvider (real),
    FakeProvider (tests)."""

    def stream_generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AsyncIterator[StreamEvent]: ...

    async def embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]: ...
