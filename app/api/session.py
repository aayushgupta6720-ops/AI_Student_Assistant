"""Sessions issued by the server, in a cookie the page's scripts can't read.

A session id unlocks that visitor's private notes and chat history, so the
client doesn't get to choose it. It used to: any 1-64 character string was
accepted (the README's curl example used "cli", so everyone who copied it
shared one set of notes), it travelled in URLs, and it was logged in full.
Now the server makes a random id the first time a browser or client calls
it, sends it in an HttpOnly, SameSite=Strict cookie, and ignores any id that
isn't one it could have made. Logs carry a hash of it, never the id.

A middleware rather than a route dependency: FastAPI doesn't copy cookies
set by a dependency onto a StreamingResponse, and /chat streams."""

import hashlib
import re
import secrets
from http.cookies import SimpleCookie

from fastapi import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

COOKIE = "sid"
_VALID = re.compile(r"[A-Za-z0-9_-]{32}")  # what secrets.token_urlsafe(24) makes


def _from_cookie(scope: Scope) -> str | None:
    for name, value in scope["headers"]:
        if name == b"cookie":
            morsel = SimpleCookie(value.decode("latin-1")).get(COOKIE)
            if morsel and _VALID.fullmatch(morsel.value):
                return morsel.value
    return None


class SessionCookie:
    """Puts the visitor's session id in scope["state"]["session_id"], and
    sets the cookie on the response when it had to make a new one."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        session = _from_cookie(scope)
        issued = session is None
        if issued:
            session = secrets.token_urlsafe(24)
        scope.setdefault("state", {})["session_id"] = session

        async def send_with_cookie(message: Message) -> None:
            if issued and message["type"] == "http.response.start":
                secure = "; Secure" if scope.get("scheme") == "https" else ""
                cookie = f"{COOKIE}={session}; Path=/; HttpOnly; SameSite=Strict{secure}"
                message["headers"] = [*message.get("headers", []), (b"set-cookie", cookie.encode())]
            await send(message)

        await self.app(scope, receive, send_with_cookie)


def session_id(request: Request) -> str:
    """The calling visitor's session (a route dependency)."""
    return request.state.session_id


def session_tag(session: str) -> str:
    """What the logs say instead of the session id: enough to follow one
    visitor's requests, not enough to act as them."""
    return hashlib.sha256(session.encode()).hexdigest()[:12]
