"""Per-visitor request limits at the HTTP edge, so one visitor can't use up
the shared daily Gemini quota for everyone. A visitor is an IP address (for
IPv6, its /64, since one visitor usually holds all 2^64 of those).

With REDIS_URL set, counts live in Redis, so they survive restarts (on
Render's free plan, the service restarts every time it wakes from its
spin-down after about 15 idle minutes) and would be shared by several
instances. Without it, or while Redis is unreachable, they live in the
process."""

import ipaddress
import math
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, HTTPException, Request
from redis import Redis
from redis.exceptions import RedisError

from app.config import Settings, get_settings
from app.observability import log_event


@dataclass(frozen=True)
class Limit:
    count: int
    window_s: float
    per: str  # how the window reads in a message: "a minute", "a day"


class RateLimiter:
    """Sliding-window limits per key. A key keeps the times and costs of its
    recent requests, never more than its largest limit allows, and at most
    max_keys keys are tracked (least recently seen dropped first), so memory
    stays bounded."""

    def __init__(
        self,
        what: str,
        limits: list[Limit],
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.what = what  # what's being counted, for the refusal: "messages"
        self.limits = [limit for limit in limits if limit.count > 0]  # 0 turns one off
        self.max_keys = max_keys
        self._clock = clock
        self._hits: OrderedDict[str, deque[tuple[float, int]]] = OrderedDict()  # (time, cost)

    def hit(self, key: str, cost: int = 1) -> tuple[Limit, float] | None:
        """Count a request from `key` costing `cost` (1 per request, or e.g.
        an upload's chunks). If it would take `key` over one of the limits,
        count nothing and return that limit and the seconds until enough of
        the earlier use has left the window for this request to fit."""
        if not self.limits:
            return None
        now = self._clock()
        hits = self._hits.pop(key, deque())
        longest = max(limit.window_s for limit in self.limits)
        while hits and now - hits[0][0] >= longest:
            hits.popleft()
        self._hits[key] = hits  # re-inserted last: the most recently seen key

        for limit in self.limits:
            in_window = [(t, c) for t, c in hits if now - t < limit.window_s]
            used = sum(c for _, c in in_window)
            if used + cost > limit.count:
                for t, c in in_window:  # oldest first, as they leave the window
                    used -= c
                    if used + cost <= limit.count:
                        return limit, t + limit.window_s - now
                return limit, limit.window_s  # costs more than the limit allows at all

        hits.append((now, cost))
        while len(self._hits) > self.max_keys:
            self._hits.popitem(last=False)
        return None


# RateLimiter.hit, run inside Redis so the check and the count are one atomic
# step. KEYS[1] is the key's sorted set of requests, scored by time, each
# member ending in ":<cost>". ARGV is now, a unique member, the cost, then
# count and window for each limit. Returns {index of the refusing limit
# (1-based, 0 if none), wait in seconds}; the wait is a string because Redis
# truncates Lua numbers to integers.
_SLIDING_WINDOW = """
local now, member, cost = tonumber(ARGV[1]), ARGV[2], tonumber(ARGV[3])
local longest = 0
for i = 4, #ARGV, 2 do
  longest = math.max(longest, tonumber(ARGV[i + 1]))
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - longest)
for i = 4, #ARGV, 2 do
  local count, window = tonumber(ARGV[i]), tonumber(ARGV[i + 1])
  local hits = redis.call('ZRANGEBYSCORE', KEYS[1], '(' .. (now - window), '+inf', 'WITHSCORES')
  local used = 0
  for j = 1, #hits, 2 do
    used = used + tonumber(string.match(hits[j], ':(%d+)$'))
  end
  if used + cost > count then
    for j = 1, #hits, 2 do  -- oldest first, as they leave the window
      used = used - tonumber(string.match(hits[j], ':(%d+)$'))
      if used + cost <= count then
        return {(i - 2) / 2, tostring(tonumber(hits[j + 1]) + window - now)}
      end
    end
    return {(i - 2) / 2, tostring(window)}
  end
end
redis.call('ZADD', KEYS[1], now, member)
redis.call('EXPIRE', KEYS[1], math.ceil(longest))
return {0, '0'}
"""


class RedisOutage:
    """After a Redis error, skip Redis for a while instead of waiting out its
    timeout on every request; log once per outage, not once per request."""

    def __init__(self, retry_after_s: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.retry_after_s = retry_after_s
        self._clock = clock
        self._down_since: float | None = None

    def skip(self) -> bool:
        return self._down_since is not None and self._clock() - self._down_since < self.retry_after_s

    def failed(self, exc: Exception) -> None:
        if self._down_since is None:
            log_event(event="redis_unavailable", used_by="rate_limit", error=str(exc))
        self._down_since = self._clock()

    def worked(self) -> None:
        self._down_since = None


class RedisRateLimiter:
    """RateLimiter's limits, counted in Redis. If Redis fails, requests are
    counted by the in-process `fallback` instead: still limited, just not kept,
    rather than either refusing everyone or letting everyone through.

    The client is synchronous because charge() is called from synchronous
    callbacks (before each model call, on each saved note). On Render's private
    network a call takes about a millisecond, and the client's short timeouts
    plus RedisOutage bound how long a Redis problem can hold up a request."""

    def __init__(
        self,
        fallback: RateLimiter,
        redis: Redis,
        outage: RedisOutage,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.what = fallback.what
        self.limits = fallback.limits
        self._fallback = fallback
        self._script = redis.register_script(_SLIDING_WINDOW)
        self._outage = outage
        # Wall-clock time, unlike RateLimiter: the counts outlive the process.
        self._clock = clock

    def hit(self, key: str, cost: int = 1) -> tuple[Limit, float] | None:
        if not self.limits:
            return None
        if self._outage.skip():
            return self._fallback.hit(key, cost)
        now = self._clock()
        args = [now, f"{now}:{secrets.token_hex(4)}:{cost}", cost]
        for limit in self.limits:
            args += [limit.count, limit.window_s]
        try:
            refused, wait = self._script(keys=[f"ratelimit:{self.what}:{key}"], args=args)
        except RedisError as exc:
            self._outage.failed(exc)
            return self._fallback.hit(key, cost)
        self._outage.worked()
        if int(refused) == 0:
            return None
        return self.limits[int(refused) - 1], float(wait)


def redis_client(url: str) -> Redis:
    return Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)


def build_rate_limiters(
    settings: Settings, redis: Redis | None = None
) -> dict[str, RateLimiter | RedisRateLimiter]:
    """One limiter per costly endpoint. A chat message counts once, and each
    model call it makes counts again (one message can take up to six), per
    visitor and across all of them (key ALL), since every visitor shares one
    daily quota. An upload or re-ingest is a few embedding calls. An upload's
    chunks count too, since one file can be ~500 chunks to embed and keep in
    memory, and so does a scanned PDF, which takes a chat-model call to
    transcribe."""
    limiters = {
        "chat": RateLimiter("messages", [
            Limit(settings.chat_limit_per_minute, 60, "a minute"),
            Limit(settings.chat_limit_per_day, 24 * 3600, "a day"),
        ]),
        "model_calls": RateLimiter("model calls", [Limit(settings.chat_model_calls_per_day, 24 * 3600, "a day")]),
        "model_calls_all": RateLimiter("model calls for everyone", [
            Limit(settings.chat_model_calls_per_day_all, 24 * 3600, "a day"),
        ]),
        "upload": RateLimiter("uploads", [Limit(settings.upload_limit_per_hour, 3600, "an hour")]),
        "upload_chunks": RateLimiter("note chunks", [Limit(settings.upload_chunk_limit_per_day, 24 * 3600, "a day")]),
        "ingest": RateLimiter("re-ingests", [Limit(settings.ingest_limit_per_hour, 3600, "an hour")]),
        "transcribe": RateLimiter("scanned PDFs", [Limit(settings.transcribe_limit_per_day, 24 * 3600, "a day")]),
    }
    if redis is None:
        return limiters
    outage = RedisOutage()  # one for all of them: Redis is either up or not
    return {name: RedisRateLimiter(limiter, redis, outage) for name, limiter in limiters.items()}


_warned_missing_header = False


def client_key(request: Request) -> str:
    """Who's asking. The connecting address, unless CLIENT_IP_HEADER names a
    header that a trusted proxy in front of the app sets (and overwrites if the
    client sent one). Never X-Forwarded-For: proxies append to it, so its first
    entry is whatever the visitor chose."""
    global _warned_missing_header
    header = get_settings().client_ip_header
    ip = request.headers.get(header, "").strip() if header else ""
    if header and not ip:
        # Everyone without the header shares one count: strict, but safe.
        # Falling back to the connecting address isn't, since uvicorn may
        # already have replaced it with X-Forwarded-For's first entry (Render
        # sets FORWARDED_ALLOW_IPS=*, which tells it to).
        if not _warned_missing_header:
            _warned_missing_header = True
            log_event(event="client_ip_header_missing", header=header)
        return f"missing {header}"
    ip = ip or (request.client.host if request.client else "unknown")
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((addr, 64), strict=False))
    return str(addr)


def _duration(seconds: int) -> str:
    if seconds < 90:
        n, unit = seconds, "second"
    elif seconds < 90 * 60:
        n, unit = math.ceil(seconds / 60), "minute"
    else:
        n, unit = math.ceil(seconds / 3600), "hour"
    return f"{n} {unit}{'' if n == 1 else 's'}"


ALL = "everyone"  # the key for limits shared by every visitor


def charge(request: Request, name: str, cost: int = 1, key: str | None = None) -> None:
    """Count `cost` against this visitor's `name` limits (or `key`'s, e.g.
    ALL), or refuse the request with a 429 if that would take them over one."""
    limiter: RateLimiter | RedisRateLimiter = request.app.state.rate_limiters[name]
    key = key or client_key(request)
    refused = limiter.hit(key, cost)
    if refused is None:
        return
    limit, retry_after_s = refused
    wait = max(1, math.ceil(retry_after_s))
    log_event(event="rate_limited", limiter=name, client=key, cost=cost, retry_after_s=wait)
    if cost == 1:
        reason = f"You've reached the limit of {limit.count} {limiter.what} {limit.per}."
    else:
        reason = (
            f"That's {cost:,} {limiter.what}, which would take you over the limit of "
            f"{limit.count:,} {limiter.what} {limit.per}."
        )
    raise HTTPException(
        status_code=429,
        detail=f"{reason} Try again in {_duration(wait)}.",
        headers={"Retry-After": str(wait)},
    )


def rate_limit(name: str):
    """A route dependency that refuses the request with a 429 once this
    visitor is over the `name` limits."""

    def check(request: Request) -> None:
        charge(request, name)

    return Depends(check)
