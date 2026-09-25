import json
from datetime import datetime, timedelta, timezone

import httpx

import app.main as main
from app.api.ratelimit import build_rate_limiters
from app.config import get_settings
from app.inference.provider import ModelOverloadedError, ModelTimeoutError, QuotaExceededError


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


async def test_timeout_says_the_request_was_stopped(monkeypatch):
    [(event, data)] = await _chat_events(monkeypatch, QuotaAgent(ModelTimeoutError("no response from Gemini within 60s")))

    assert event == "error"
    assert data["kind"] == "timeout"
    assert "didn't respond within 60 seconds" in data["message"]
