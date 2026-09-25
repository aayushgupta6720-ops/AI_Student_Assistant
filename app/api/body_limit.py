"""Refuses request bodies over a size limit before the app reads them.

FastAPI reads and parses the whole body before validating any field, so a
field's max_length doesn't help here: one 52 MB request cost ~250 MB of
memory, and Cloudflare in front of Render lets bodies up to 100 MB through,
more than a free instance's 512 MB can take."""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodySizeLimit:
    def __init__(self, app: ASGIApp, max_bytes: int, max_bytes_by_path: dict[str, int] | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.max_bytes_by_path = max_bytes_by_path or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.max_bytes_by_path.get(scope["path"], self.max_bytes)

        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            await _too_large(send, limit)  # refused without reading a byte of it
            return

        # No Content-Length (chunked), or one that may understate: read up to
        # the limit here, then hand the app what was read.
        received: list[Message] = []
        size = 0
        while True:
            message = await receive()
            received.append(message)
            if message["type"] != "http.request":
                break  # the client went away
            size += len(message.get("body", b""))
            if size > limit:
                await _too_large(send, limit)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> Message:
            # What was read first, then the live channel, so a streaming
            # response still hears about a client disconnect.
            return received.pop(0) if received else await receive()

        await self.app(scope, replay, send)


def _readable(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB" if n >= 1024 * 1024 else f"{n // 1024} KB"


async def _too_large(send: Send, limit: int) -> None:
    body = json.dumps({"detail": f"The request is over the {_readable(limit)} limit."}).encode()
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
    })
    await send({"type": "http.response.body", "body": body})
