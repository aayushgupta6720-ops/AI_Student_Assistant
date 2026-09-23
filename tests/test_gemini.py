from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError

import app.inference.gemini as gemini


def _rate_limited(retry_delay: str) -> ClientError:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
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
    "error",
    [
        _rate_limited("59s"),  # a daily quota: waiting a minute won't help
        ClientError(400, {"error": {"code": 400, "message": "bad request"}}),
    ],
)
async def test_embed_does_not_retry_what_retrying_cannot_fix(embed_calls, error):
    calls = embed_calls([error])

    with pytest.raises(ClientError):
        await gemini.GeminiProvider().embed(["a"], "query")
    assert len(calls) == 1
