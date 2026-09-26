import json
from datetime import datetime, timedelta, timezone

import httpx

import app.api.routes as routes
import app.main as main
from app.api.ratelimit import build_rate_limiters
from app.config import get_settings
from app.inference.provider import ModelOverloadedError, ModelTimeoutError, QuotaExceededError
from app.intelligence.agent import AgentDone
from app.observability import CallTrace, StepRecord


class QuotaAgent:
    def __init__(self, error: QuotaExceededError) -> None:
        self.error = error

    async def run_turn(self, session_id, user_text, timezone=None):
        raise self.error
        yield  # unreachable; makes this an async generator like Agent.run_turn


async def _chat_events(monkeypatch, agent) -> list[tuple[str, dict]]:
    monkeypatch.setattr(main.app.state, "agent", agent, raising=False)
    monkeypatch.setattr(main.app.state, "rate_limiters", build_rate_limiters(get_settings()), raising=False)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json={"session_id": "s", "message": "hi"})
    events = []
    for block in response.text.strip().split("\n\n"):
        event, data = block.split("\n", 1)
        events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


async def test_daily_quota_shows_a_readable_message_not_the_raw_429(monkeypatch):
    resets_at = datetime(2026, 9, 25, tzinfo=timezone(timedelta(hours=-7)))
    error = QuotaExceededError("429 RESOURCE_EXHAUSTED. {...}", daily=True, resets_at=resets_at)

    [(event, data)] = await _chat_events(monkeypatch, QuotaAgent(error))

    assert event == "error"
    assert data["kind"] == "quota"
    assert "used up today's free Gemini quota" in data["message"]
    assert "RESOURCE_EXHAUSTED" not in data["message"]
    assert data["resets_at"] == "2026-09-25T00:00:00-07:00"


async def test_per_minute_quota_says_to_wait_a_minute(monkeypatch):
    error = QuotaExceededError("429 RESOURCE_EXHAUSTED", daily=False)

    [(_, data)] = await _chat_events(monkeypatch, QuotaAgent(error))

    assert "Wait a minute" in data["message"]
    assert data["resets_at"] is None


async def test_overloaded_model_says_it_is_temporary(monkeypatch):
    [(event, data)] = await _chat_events(monkeypatch, QuotaAgent(ModelOverloadedError("503 UNAVAILABLE. {...}")))

    assert event == "error"
    assert data["kind"] == "overloaded"
    assert "Gemini is overloaded right now" in data["message"]
    assert "UNAVAILABLE" not in data["message"]


async def test_an_unexpected_error_shows_a_plain_message_not_the_exception(monkeypatch):
    # The chat used to show "ClientError: 400 INVALID_ARGUMENT. {...}", the
    # provider's whole error body.
    error = RuntimeError("400 INVALID_ARGUMENT. {'error': {'message': 'Invalid JSON payload received.'}}")

    [(event, data)] = await _chat_events(monkeypatch, QuotaAgent(error))

    assert event == "error" and data["kind"] == "internal"
    assert "Something went wrong" in data["message"]
    assert "INVALID_ARGUMENT" not in data["message"] and "RuntimeError" not in data["message"]


class SavingAgent:
    """A turn that saved a private note, as its trace records it."""

    async def run_turn(self, session_id, user_text, timezone=None):
        trace = CallTrace()
        trace.add(StepRecord("tools", "save_note", 1.0, 1.0, depth=1,
                             meta={"args": {"title": "Bank", "content": "PIN 4321"}}))
        yield AgentDone(answer="Saved.", sources=["bank"], iterations=2, trace=trace, tools_used=["save_note"])


async def test_the_log_keeps_the_length_of_what_was_typed_not_the_text(monkeypatch):
    # Messages and tool arguments can hold a private note, and the log
    # outlives it by far.
    logged = []
    monkeypatch.setattr(routes, "log_event", lambda **fields: logged.append(fields))

    events = dict(await _chat_events(monkeypatch, SavingAgent()))

    [call] = [f for f in logged if f["event"] == "chat_call"]
    assert call["query"] == "<2 chars>"
    assert call["steps"][0]["meta"] == {"args": {"title": "<4 chars>", "content": "<8 chars>"}}
    assert "PIN" not in json.dumps(call) and call["tools_used"] == ["save_note"]
    assert events["done"]["steps"][0]["meta"]["args"]["content"] == "PIN 4321"  # the chat still sees it all


async def test_log_chat_text_logs_the_text_for_debugging(monkeypatch):
    monkeypatch.setattr(get_settings(), "log_chat_text", True)
    logged = []
    monkeypatch.setattr(routes, "log_event", lambda **fields: logged.append(fields))

    await _chat_events(monkeypatch, SavingAgent())

    [call] = [f for f in logged if f["event"] == "chat_call"]
    assert call["query"] == "hi" and call["steps"][0]["meta"]["args"]["content"] == "PIN 4321"


async def test_timeout_says_the_request_was_stopped(monkeypatch):
    [(event, data)] = await _chat_events(monkeypatch, QuotaAgent(ModelTimeoutError("no response from Gemini within 60s")))

    assert event == "error"
    assert data["kind"] == "timeout"
    assert "didn't respond within 60 seconds" in data["message"]
