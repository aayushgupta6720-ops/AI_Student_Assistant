"""GeminiProvider - the ONLY module in the app that imports google.genai.

Responsibilities:
  * translate neutral Message/ToolSpec -> Gemini Content/Part/Tool
  * stream the model's output and translate it back into neutral StreamEvents
  * embeddings
  * retry on 429 using the server's RetryInfo delay
"""

import asyncio
import re
import uuid
from datetime import datetime, timedelta
from functools import lru_cache
from typing import AsyncIterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from app.config import get_settings
from app.inference.provider import EmbedKind, ModelOverloadedError, ModelTimeoutError, QuotaExceededError
from app.inference.types import (
    Message,
    StreamEnd,
    StreamEvent,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
    Usage,
)

_MAX_RATE_LIMIT_RETRIES = 5
# A 503 "high demand" usually clears within seconds to minutes: retry a few
# times (1s, 2s, 4s) rather than make the user wait out a long spike.
_MAX_OVERLOAD_ATTEMPTS = 4
_OVERLOAD_BASE_DELAY_S = 1.0
_EMBED_TASK = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}
# Gemini rejects an embed request with more than 100 texts ("at most 100
# requests can be in one batch").
MAX_TEXTS_PER_EMBED_REQUEST = 100


@lru_cache
def _client() -> genai.Client:
    settings = get_settings()
    return genai.Client(
        api_key=settings.gemini_api_key,
        # Milliseconds, applied to connecting and to each read, so a stream
        # that stalls partway is caught as well as one that never starts.
        http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_s * 1000)),
    )


def _timeout_error() -> ModelTimeoutError:
    # Not retried: a retry would double an already long wait.
    return ModelTimeoutError(f"no response from Gemini within {get_settings().gemini_timeout_s:g}s")


def _rate_limit_retry_delay(exc: ClientError, fallback: float) -> float:
    details = (exc.details or {}).get("error", {}).get("details", [])
    for detail in details:
        if detail.get("@type", "").endswith("RetryInfo"):
            match = re.match(r"([\d.]+)s?", detail.get("retryDelay", ""))
            if match:
                return float(match.group(1))
    return fallback


def _is_daily_quota(exc: ClientError) -> bool:
    """True for a 429 caused by a per-day quota (e.g. the free tier's
    requests-per-day cap). Its RetryInfo still suggests ~60s, but no retry can
    succeed until the quota resets."""
    details = (exc.details or {}).get("error", {}).get("details", [])
    return any(
        "PerDay" in violation.get("quotaId", "")
        for detail in details
        if detail.get("@type", "").endswith("QuotaFailure")
        for violation in detail.get("violations", [])
    )


def _next_pacific_midnight() -> datetime | None:
    """When Gemini's per-day quotas next reset: midnight Pacific time."""
    try:
        pacific = ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        return None  # no tz database on this host; callers still say "midnight Pacific"
    tomorrow = datetime.now(pacific).date() + timedelta(days=1)
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=pacific)


def _overload_backoff(exc: ServerError, attempt: int) -> float:
    """Seconds to wait before retrying a 503 UNAVAILABLE ("high demand"),
    which Google says is usually temporary. Re-raises any other server error,
    and raises ModelOverloadedError once the retries run out."""
    if exc.code != 503:
        raise exc
    if attempt >= _MAX_OVERLOAD_ATTEMPTS:
        raise ModelOverloadedError(str(exc)) from exc
    return _OVERLOAD_BASE_DELAY_S * 2 ** (attempt - 1)


def _rate_limit_backoff(exc: ClientError, attempt: int) -> float:
    """Seconds to wait before retrying exc. Raises instead when retrying isn't
    worth it: exc itself if it isn't a 429, QuotaExceededError for a 429 we
    can't wait out (a daily quota, out of attempts, or the server wants a
    longer wait than rate_limit_max_wait_s)."""
    if exc.code != 429:
        raise exc
    if _is_daily_quota(exc):
        raise QuotaExceededError(str(exc), daily=True, resets_at=_next_pacific_midnight()) from exc
    delay = _rate_limit_retry_delay(exc, fallback=2**attempt)
    if attempt == _MAX_RATE_LIMIT_RETRIES or delay > get_settings().rate_limit_max_wait_s:
        raise QuotaExceededError(str(exc), daily=False) from exc
    return delay


# ---- neutral -> Gemini -------------------------------------------------------


def _to_gemini_contents(messages: list[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for msg in messages:
        parts: list[types.Part] = []
        for part in msg.parts:
            if isinstance(part, TextPart):
                if part.text:
                    parts.append(
                        types.Part(
                            text=part.text,
                            thought_signature=part.provider_state.get("thought_signature"),
                        )
                    )
            elif isinstance(part, ToolCallPart):
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=part.id, name=part.name, args=part.args
                        ),
                        # Gemini 3 rejects replayed function calls without this.
                        thought_signature=part.provider_state.get("thought_signature"),
                    )
                )
            elif isinstance(part, ToolResultPart):
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=part.call_id, name=part.name, response=part.result
                        )
                    )
                )
        if not parts:
            continue
        # Gemini has two roles: "user" and "model". Tool results go back as "user".
        role = "model" if msg.role == "assistant" else "user"
        contents.append(types.Content(role=role, parts=parts))
    return contents


def _to_gemini_tools(tools: list[ToolSpec]) -> list[types.Tool] | None:
    if not tools:
        return None
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.input_schema,
                )
                for t in tools
            ]
        )
    ]


class GeminiProvider:
    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.generation_model
        self.embedding_model = settings.embedding_model
        self.embedding_dim = settings.embedding_dim

    async def stream_generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AsyncIterator[StreamEvent]:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=_to_gemini_tools(tools),
            # We run the tool loop ourselves in the intelligence layer.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = _to_gemini_contents(messages)

        # The HTTP request is only sent when the first chunk is pulled, so the
        # 429 retry has to wrap that first pull, not the stream construction.
        for attempt in range(1, _MAX_RATE_LIMIT_RETRIES + 1):
            try:
                stream = await _client().aio.models.generate_content_stream(
                    model=self.model, contents=contents, config=config
                )
                first = await anext(stream)
                break
            except StopAsyncIteration:
                yield StreamEnd(finish_reason="stop")
                return
            except ClientError as exc:
                await asyncio.sleep(_rate_limit_backoff(exc, attempt))
            except ServerError as exc:
                await asyncio.sleep(_overload_backoff(exc, attempt))
            except httpx.TimeoutException as exc:
                raise _timeout_error() from exc

        async def _chunks():
            yield first
            try:
                async for c in stream:
                    yield c
            except httpx.TimeoutException as exc:  # the stream stalled partway
                raise _timeout_error() from exc

        finish_reason = "stop"
        usage: Usage | None = None
        async for chunk in _chunks():
            if chunk.usage_metadata is not None:
                meta = chunk.usage_metadata
                usage = Usage(
                    input_tokens=meta.prompt_token_count or 0,
                    output_tokens=(meta.candidates_token_count or 0)
                    + (meta.thoughts_token_count or 0),
                )
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if candidate.finish_reason:
                finish_reason = str(candidate.finish_reason.name).lower()
            for part in candidate.content.parts or []:
                state = (
                    {"thought_signature": part.thought_signature}
                    if part.thought_signature
                    else {}
                )
                if part.function_call:
                    fc = part.function_call
                    yield ToolCall(
                        id=fc.id or f"call_{uuid.uuid4().hex[:8]}",
                        name=fc.name,
                        args=dict(fc.args or {}),
                        provider_state=state,
                    )
                elif part.text and not getattr(part, "thought", False):
                    yield TextDelta(text=part.text, provider_state=state)

        if usage is not None:
            yield usage
        yield StreamEnd(finish_reason=finish_reason)

    async def embed(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        vectors: list[list[float]] = []
        # Sequential, not concurrent: parallel batches would just trip the
        # per-minute quota sooner.
        for start in range(0, len(texts), MAX_TEXTS_PER_EMBED_REQUEST):
            batch = texts[start : start + MAX_TEXTS_PER_EMBED_REQUEST]
            vectors.extend(await self._embed_batch(batch, kind))
        return vectors

    async def _embed_batch(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=_EMBED_TASK[kind],
            output_dimensionality=self.embedding_dim,
        )
        for attempt in range(1, _MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = await _client().aio.models.embed_content(
                    model=self.embedding_model, contents=texts, config=config
                )
                return [e.values for e in response.embeddings]
            except ClientError as exc:
                await asyncio.sleep(_rate_limit_backoff(exc, attempt))
            except ServerError as exc:
                await asyncio.sleep(_overload_backoff(exc, attempt))
            except httpx.TimeoutException as exc:
                raise _timeout_error() from exc
        raise AssertionError("unreachable")


@lru_cache
def get_provider() -> GeminiProvider:
    return GeminiProvider()
