import httpx
import pytest
from starlette.requests import Request

import app.api.ratelimit as ratelimit
import app.main as main
from app.api.ratelimit import Limit, RateLimiter, build_rate_limiters, client_key
from app.config import get_settings


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ---- the limiter -----------------------------------------------------------------


def test_allows_up_to_the_limit_then_says_when_to_retry():
    clock = Clock()
    limiter = RateLimiter("messages", [Limit(3, 60, "a minute")], clock=clock)
    for _ in range(3):
        assert limiter.hit("a") is None
        clock.now += 10

    limit, retry_after = limiter.hit("a")
    assert limit.count == 3 and retry_after == pytest.approx(30)  # when the first one leaves the window
    clock.now += 30
    assert limiter.hit("a") is None


def test_refused_requests_do_not_count_so_hammering_does_not_extend_the_wait():
    clock = Clock()
    limiter = RateLimiter("messages", [Limit(1, 60, "a minute")], clock=clock)
    limiter.hit("a")
    for _ in range(50):
        clock.now += 1
        assert limiter.hit("a") is not None
    clock.now += 10
    assert limiter.hit("a") is None


def test_every_limit_applies_and_visitors_are_counted_separately():
    clock = Clock()
    limiter = RateLimiter("messages", [Limit(2, 60, "a minute"), Limit(3, 86400, "a day")], clock=clock)
    assert limiter.hit("a") is None and limiter.hit("a") is None
    assert limiter.hit("a")[0].per == "a minute"
    clock.now += 61
    assert limiter.hit("a") is None
    assert limiter.hit("a")[0].per == "a day"
    assert limiter.hit("b") is None  # someone else


def test_a_cost_counts_against_the_limit_and_the_wait_is_until_it_fits():
    clock = Clock()
    limiter = RateLimiter("note chunks", [Limit(1000, 86400, "a day")], clock=clock)
    assert limiter.hit("a", 600) is None
    clock.now += 3600
    assert limiter.hit("a", 300) is None

    limit, retry_after = limiter.hit("a", 200)  # 900 used: fits once the 600 leave the window
    assert limit.count == 1000 and retry_after == pytest.approx(86400 - 3600)
    assert limiter.hit("a", 100) is None  # a smaller one still fits now
    assert limiter.hit("a", 1001)[1] == 86400  # never fits: the longest wait there is


def test_a_zero_limit_is_off_and_tracked_visitors_are_capped():
    assert RateLimiter("x", [Limit(0, 60, "a minute")]).hit("a") is None
    limiter = RateLimiter("x", [Limit(5, 60, "a minute")], max_keys=2)
    for key in ("a", "b", "c"):
        limiter.hit(key)
    assert list(limiter._hits) == ["b", "c"]


# ---- who the visitor is -------------------------------------------------------------


def _request(peer: str, **headers: str) -> Request:
    raw = [(k.replace("_", "-").lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw, "client": (peer, 1234)})


def test_by_default_the_visitor_is_the_connecting_address_and_forwarded_headers_are_ignored(monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", None)
    spoofed = _request("203.0.113.7", x_forwarded_for="1.1.1.1", cf_connecting_ip="2.2.2.2")
    assert client_key(spoofed) == "203.0.113.7"


def test_behind_a_trusted_proxy_the_configured_header_names_the_visitor(monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", "CF-Connecting-IP")
    request = _request("10.0.0.1", cf_connecting_ip="198.51.100.4", x_forwarded_for="1.1.1.1")
    assert client_key(request) == "198.51.100.4"


def test_without_the_proxy_header_everyone_shares_one_count_and_it_is_logged_once(monkeypatch):
    # Not the connecting address: behind Render, uvicorn may have set that from
    # X-Forwarded-For, which the visitor controls.
    monkeypatch.setattr(get_settings(), "client_ip_header", "CF-Connecting-IP")
    monkeypatch.setattr(ratelimit, "_warned_missing_header", False)
    logged = []
    monkeypatch.setattr(ratelimit, "log_event", lambda **fields: logged.append(fields))

    keys = {client_key(_request(peer)) for peer in ("10.9.9.1", "10.9.9.2")}

    assert keys == {"missing CF-Connecting-IP"}
    assert logged == [{"event": "client_ip_header_missing", "header": "CF-Connecting-IP"}]


@pytest.mark.parametrize(
    ("ip", "key"),
    [("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"), ("2001:db8:1:2:ffff::9", "2001:db8:1:2::/64"),
     ("::ffff:203.0.113.7", "203.0.113.7"), ("not-an-ip", "not-an-ip")],
)
def test_ipv6_visitors_are_grouped_by_their_64_block(monkeypatch, ip, key):
    monkeypatch.setattr(get_settings(), "client_ip_header", "CF-Connecting-IP")
    assert client_key(_request("10.0.0.1", cf_connecting_ip=ip)) == key


# ---- through the API ------------------------------------------------------------------


class CountingAgent:
    def __init__(self) -> None:
        self.turns = 0

    async def run_turn(self, session_id, user_text, timezone=None):
        self.turns += 1
        return
        yield  # an async generator, like Agent.run_turn


@pytest.fixture
def agent(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "client_ip_header", "CF-Connecting-IP")
    monkeypatch.setattr(settings, "chat_limit_per_minute", 2)
    agent = CountingAgent()
    monkeypatch.setattr(main.app.state, "agent", agent, raising=False)
    monkeypatch.setattr(main.app.state, "rate_limiters", build_rate_limiters(settings), raising=False)
    return agent


async def _chat(visitor: str, session: str = "s", headers: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=main.app, client=("10.0.0.1", 1234))  # Render's proxy
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/chat",
            json={"session_id": session, "message": "hi"},
            headers={"CF-Connecting-IP": visitor, **(headers or {})},
        )


async def test_a_visitor_over_the_chat_limit_gets_a_429_and_the_model_is_not_called(agent):
    assert [(await _chat("198.51.100.4")).status_code for _ in range(2)] == [200, 200]

    refused = await _chat("198.51.100.4")

    assert refused.status_code == 429 and agent.turns == 2
    assert int(refused.headers["Retry-After"]) == 60
    assert refused.json()["detail"] == "You've reached the limit of 2 messages a minute. Try again in 60 seconds."


async def test_other_visitors_are_unaffected_and_new_sessions_or_forged_headers_do_not_reset_the_count(agent):
    for _ in range(2):
        await _chat("198.51.100.4")

    assert (await _chat("198.51.100.4", session="new-tab", headers={"X-Forwarded-For": "1.2.3.4"})).status_code == 429
    assert (await _chat("203.0.113.9")).status_code == 200
    assert agent.turns == 3
