from datetime import datetime
from typing import AsyncIterator, Literal, Protocol

from app.inference.types import Message, StreamEvent, ToolSpec

EmbedKind = Literal["document", "query"]


class ModelOverloadedError(Exception):
    """The model stayed overloaded (a provider-side capacity spike) through
    our retries. Nothing is wrong with the request; trying again in a minute
    usually works."""


class QuotaExceededError(Exception):
    """The provider refused a call over quota and retrying now won't help.
    Provider-neutral, so the API layer can tell the user what happened
    without knowing which vendor is behind it."""

    def __init__(self, message: str, *, daily: bool, resets_at: datetime | None = None) -> None:
        super().__init__(message)
        # True for a per-day quota: nothing works again until it resets.
        self.daily = daily
        self.resets_at = resets_at


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
