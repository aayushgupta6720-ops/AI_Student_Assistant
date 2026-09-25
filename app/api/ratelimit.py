"""Per-visitor request limits at the HTTP edge, so one visitor can't use up
the shared daily Gemini quota for everyone. A visitor is an IP address (for
IPv6, its /64, since one visitor usually holds all 2^64 of those).

Counts live in memory: right for a single instance (Render's free plan runs
one), and they reset on restart. Several instances would need a shared
store such as Redis."""

import ipaddress
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, HTTPException, Request

from app.config import Settings, get_settings
from app.observability import log_event


@dataclass(frozen=True)
class Limit:
    count: int
    window_s: float
    per: str  # how the window reads in a message: "a minute", "a day"


class RateLimiter:
    """Sliding-window limits per key. A key keeps the times of its recent
    requests, never more than its largest limit allows, and at most
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
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def hit(self, key: str) -> tuple[Limit, float] | None:
        """Count a request from `key`. If one of the limits is already
        reached, count nothing and return that limit and the seconds until it
        allows another request."""
        if not self.limits:
            return None
        now = self._clock()
        hits = self._hits.pop(key, deque())
        longest = max(limit.window_s for limit in self.limits)
        while hits and now - hits[0] >= longest:
            hits.popleft()
        self._hits[key] = hits  # re-inserted last: the most recently seen key

        for limit in self.limits:
            in_window = [t for t in hits if now - t < limit.window_s]
            if len(in_window) >= limit.count:
                return limit, in_window[-limit.count] + limit.window_s - now

        hits.append(now)
        while len(self._hits) > self.max_keys:
            self._hits.popitem(last=False)
        return None


def build_rate_limiters(settings: Settings) -> dict[str, RateLimiter]:
    """One limiter per costly endpoint: a chat turn is 2-3 model calls, and an
    upload or re-ingest is a few embedding calls."""
    return {
        "chat": RateLimiter("messages", [
            Limit(settings.chat_limit_per_minute, 60, "a minute"),
            Limit(settings.chat_limit_per_day, 24 * 3600, "a day"),
        ]),
        "upload": RateLimiter("uploads", [Limit(settings.upload_limit_per_hour, 3600, "an hour")]),
        "ingest": RateLimiter("re-ingests", [Limit(settings.ingest_limit_per_hour, 3600, "an hour")]),
    }


_warned_missing_header = False


def client_key(request: Request) -> str:
    """Who's asking. The connecting address, unless CLIENT_IP_HEADER names a
    header that a trusted proxy in front of the app sets (and overwrites if the
    client sent one). Never X-Forwarded-For: proxies append to it, so its first
    entry is whatever the visitor chose."""
    global _warned_missing_header
    header = get_settings().client_ip_header
    ip = request.headers.get(header, "").strip() if header else ""
    if header and not ip and not _warned_missing_header:
        # Falls back to the proxy's address, so every visitor shares one
        # count: strict, but a sign the header setting is wrong for this host.
        _warned_missing_header = True
        log_event(event="client_ip_header_missing", header=header)
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


def rate_limit(name: str):
    """A route dependency that refuses the request with a 429 once this
    visitor is over the `name` limits."""

    def check(request: Request) -> None:
        limiter: RateLimiter = request.app.state.rate_limiters[name]
        key = client_key(request)
        refused = limiter.hit(key)
        if refused is None:
            return
        limit, retry_after_s = refused
        wait = max(1, math.ceil(retry_after_s))
        log_event(event="rate_limited", limiter=name, client=key, retry_after_s=wait)
        raise HTTPException(
            status_code=429,
            detail=f"You've reached the limit of {limit.count} {limiter.what} {limit.per}. "
            f"Try again in {_duration(wait)}.",
            headers={"Retry-After": str(wait)},
        )

    return Depends(check)
