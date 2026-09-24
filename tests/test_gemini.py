from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError, ServerError

import app.inference.gemini as gemini
from app.inference.provider import ModelOverloadedError, QuotaExceededError
from app.inference.types import Message, TextPart

DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def _rate_limited(retry_delay: str, quota_id: str | None = None) -> ClientError:
    """A 429 shaped like Gemini's: a RetryInfo delay, plus a QuotaFailure
    naming the quota that was hit when quota_id is given."""
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    if quota_id:
        details.append({
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id, "quotaValue": "500"}],
        })
    return ClientError(429, {"error": {"code": 429, "message": "quota", "details": details}})


@pytest.fixture
def embed_calls(monkeypatch):
    """Returns a setter: embed_calls(outcomes) installs a fake Gemini client
    whose embed_content plays `outcomes` in order (an exception is raised, a
    None returns one vector per text) and records each call's batch."""

    def install(outcomes):
        calls = []

        async def embed_content(*, model, contents, config):
            calls.append(list(contents))
            outcome = outcomes.pop(0) if outcomes else None
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[t]) for t in contents])

        client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content)))
        monkeypatch.setattr(gemini, "_client", lambda: client)
        return calls

    return install


async def test_embed_splits_requests_at_the_api_limit(embed_calls):
    calls = embed_calls([])
    texts = [f"t{i}" for i in range(2 * gemini.MAX_TEXTS_PER_EMBED_REQUEST + 1)]

    vectors = await gemini.GeminiProvider().embed(texts, "document")

    assert [len(c) for c in calls] == [100, 100, 1]
    assert vectors == [[t] for t in texts]  # order preserved across batches


async def test_embed_retries_a_rate_limit_then_succeeds(embed_calls):
    calls = embed_calls([_rate_limited("0.01s")])

    assert await gemini.GeminiProvider().embed(["a"], "query") == [["a"]]
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_rate_limited("0.01s", quota_id=DAILY), QuotaExceededError),  # short delay, still daily
        (_rate_limited("59s"), QuotaExceededError),  # longer than rate_limit_max_wait_s
        (ClientError(400, {"error": {"code": 400, "message": "bad request"}}), ClientError),
    ],
)
async def test_embed_does_not_retry_what_retrying_cannot_fix(embed_calls, error, expected):
    calls = embed_calls([error])

    with pytest.raises(expected):
        await gemini.GeminiProvider().embed(["a"], "query")
    assert len(calls) == 1


async def test_daily_quota_on_chat_becomes_a_neutral_error_with_the_reset_time(monkeypatch):
    async def generate_content_stream(**kwargs):
        raise _rate_limited("59s", quota_id=DAILY)

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_content_stream)))
    monkeypatch.setattr(gemini, "_client", lambda: client)

    with pytest.raises(QuotaExceededError) as caught:
        async for _ in gemini.GeminiProvider().stream_generate(
            system="s", messages=[Message("user", [TextPart("hi")])], tools=[]
        ):
            pass

    assert caught.value.daily
    resets_at = caught.value.resets_at
    assert (resets_at.hour, resets_at.minute) == (0, 0)  # midnight...
    assert resets_at.utcoffset().total_seconds() / 3600 in (-7, -8)  # ...Pacific (PDT or PST)


def _overloaded() -> ServerError:
    return ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(gemini, "_OVERLOAD_BASE_DELAY_S", 0)


async def test_embed_retries_an_overloaded_model_then_succeeds(embed_calls, no_backoff):
    calls = embed_calls([_overloaded(), _overloaded()])

    assert await gemini.GeminiProvider().embed(["a"], "query") == [["a"]]
    assert len(calls) == 3


async def test_persistent_overload_becomes_model_overloaded_error(embed_calls, no_backoff):
    calls = embed_calls([_overloaded()] * 5)

    with pytest.raises(ModelOverloadedError):
        await gemini.GeminiProvider().embed(["a"], "query")
    assert len(calls) == gemini._MAX_OVERLOAD_ATTEMPTS


async def test_other_server_errors_are_not_retried(embed_calls, no_backoff):
    calls = embed_calls([ServerError(500, {"error": {"code": 500, "message": "boom"}})])

    with pytest.raises(ServerError):
        await gemini.GeminiProvider().embed(["a"], "query")
    assert len(calls) == 1


async def test_overloaded_chat_becomes_model_overloaded_error(monkeypatch, no_backoff):
    calls = []

    async def generate_content_stream(**kwargs):
        calls.append(kwargs)
        raise _overloaded()

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_content_stream)))
    monkeypatch.setattr(gemini, "_client", lambda: client)

    with pytest.raises(ModelOverloadedError):
        async for _ in gemini.GeminiProvider().stream_generate(
            system="s", messages=[Message("user", [TextPart("hi")])], tools=[]
        ):
            pass
    assert len(calls) == gemini._MAX_OVERLOAD_ATTEMPTS


def test_overload_backoff_doubles_from_one_second():
    assert [gemini._overload_backoff(_overloaded(), n) for n in (1, 2, 3)] == [1.0, 2.0, 4.0]
