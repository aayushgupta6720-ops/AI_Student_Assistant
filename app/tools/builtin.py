"""The five built-in tools. Note which layers each one touches:

  search_notes     -> knowledge layer (which itself calls inference for embeddings)
  save_note        -> knowledge layer (stores a private note for this session)
  calculator       -> pure computation, no other layer
  current_datetime -> pure computation
  fetch_url        -> external I/O
"""

import ast
import asyncio
import contextvars
import ipaddress
import math
import operator
import re
import socket
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app.config import get_settings
from app.inference.provider import LLMProvider
from app.knowledge.ingest import slugify
from app.knowledge.retrieval import current_session, get_store, retrieve
from app.knowledge.store import VectorStore
from app.knowledge.uploads import UPLOAD_TTL_S, store_private_note
from app.tools.registry import Tool, ToolRegistry

# ---- calculator: a safe arithmetic evaluator (no eval()) --------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
# About 1,200 digits: more than any real calculation needs, and small enough
# that every step is instant. The tool runs on the event loop, so an
# unbounded 9**9**9 would stall every other chat while it computed.
_MAX_INT_BITS = 4000
# A "%" with no number or "(" after it is a percentage (17% -> 17/100);
# one between two operands is modulo (10 % 3, 10 % -3).
_PERCENT_RE = re.compile(r"%(?!\s*[-+]?[\d.(])")


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int):
            # Check before computing: the power itself is the slow part.
            # (|left| >= 2, so right alone past the limit is already too big.)
            if abs(left) > 1 and (right > _MAX_INT_BITS or right * math.log2(abs(left)) > _MAX_INT_BITS):
                raise ValueError("result too large")
        result = _BIN_OPS[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > _MAX_INT_BITS:
            raise ValueError("result too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> dict:
    cleaned = _PERCENT_RE.sub("/100", expression.replace(",", "").replace("^", "**"))
    tree = ast.parse(cleaned, mode="eval")
    return {"expression": expression, "result": _eval_node(tree)}


# ---- current_datetime -------------------------------------------------------

# The user's IANA time zone for this turn, which the web client sends with
# each message. The agent sets it; None means the server's own zone, which on
# a cloud host is usually UTC, a day off for much of the world around midnight.
user_timezone: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "user_timezone", default=None
)


def _zone(name: str | None) -> ZoneInfo | None:
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None  # unknown or malformed: fall back to the server's zone


def current_datetime() -> dict:
    zone = _zone(user_timezone.get())
    now = datetime.now(zone) if zone else datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "weekday": now.strftime("%A"),
        "timezone": str(zone) if zone else now.tzname(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---- fetch_url ----------------------------------------------------------------

# Plenty of HTML to find max_chars of text in; reading stops here so a huge
# or endless response can't fill the server's memory.
MAX_FETCH_BYTES = 2 * 1024 * 1024
MAX_FETCH_REDIRECTS = 5
FETCH_DEADLINE_S = 20.0  # for the whole fetch; httpx's timeout is per read

# A <script>/<style> tag, any other tag, or a word. None of them scans past a
# "<", so an unclosed tag stops at the next one instead of rescanning to the
# end of the page (a backtracking regex took minutes on 2 MB of <script>s).
_HTML_TOKEN_RE = re.compile(
    r"(?P<block><(?P<close>/?)(?P<name>script|style)\b[^<>]*>)|(?P<tag><[^<>]*>)|(?P<word><?[^<\s]+)",
    re.I,
)


def _visible_text(html: str, max_chars: int) -> str:
    """Up to max_chars of the page's words, minus tags, scripts and styles.
    One left-to-right pass that stops once it has enough, so its time and
    memory don't grow with the page. An unclosed <script> hides the rest of the
    page, as it does in a browser."""
    words: list[str] = []
    length, open_block = 0, None
    for m in _HTML_TOKEN_RE.finditer(html):
        if m["block"]:
            name = m["name"].lower()
            if open_block is None and not m["close"]:
                open_block = name
            elif open_block == name and m["close"]:
                open_block = None
        elif m["word"] and open_block is None:
            words.append(m["word"])
            length += len(m["word"]) + 1
            if length > max_chars:
                break
    return " ".join(words)[:max_chars]


def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%", 1)[0])  # drop an IPv6 zone id
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global


def _refuse_private(ip: str, host: str) -> None:
    if not _is_public(ip):
        raise ValueError(f"{host} is a private or local address; only public web pages can be fetched")


async def _check_public_url(url: httpx.URL) -> None:
    """Refuse anything but http(s) to public addresses. The server can reach
    places a visitor can't (itself, its host's private network, cloud metadata
    endpoints), and fetch_url must not become a way in."""
    if url.scheme not in ("http", "https") or not url.host:
        raise ValueError("only absolute http(s) URLs can be fetched")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(url.host, url.port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"couldn't resolve {url.host}") from exc
    for *_, sockaddr in infos:
        _refuse_private(sockaddr[0], url.host)


def _check_connected_peer(response: httpx.Response, host: str) -> None:
    """Check the address actually connected to as well, in case DNS answered
    differently from when _check_public_url asked (DNS rebinding)."""
    stream = response.extensions.get("network_stream")
    peer = stream.get_extra_info("server_addr") if stream else None
    if peer:
        _refuse_private(peer[0], host)


async def _read_capped(response: httpx.Response) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) >= MAX_FETCH_BYTES:
            break
    del body[MAX_FETCH_BYTES:]
    return bytes(body)


async def _get_public_page(url: httpx.URL) -> tuple[httpx.Response, bytes]:
    # Redirects are followed by hand so every hop gets the address check, and
    # trust_env=False connects directly, so the peer checked is the page's
    # own server rather than a proxy.
    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        for _ in range(MAX_FETCH_REDIRECTS + 1):
            await _check_public_url(url)
            async with client.stream("GET", url, headers={"User-Agent": "AI_Student_Assistant/0.1"}) as response:
                _check_connected_peer(response, url.host)
                if response.next_request is not None:
                    url = response.next_request.url
                    continue
                response.raise_for_status()
                return response, await _read_capped(response)
    raise ValueError(f"more than {MAX_FETCH_REDIRECTS} redirects")


async def fetch_url(url: str, max_chars: int = 4000) -> dict:
    try:
        async with asyncio.timeout(FETCH_DEADLINE_S):
            response, body = await _get_public_page(httpx.URL(url))
    except TimeoutError as exc:
        raise TimeoutError(f"gave up after {FETCH_DEADLINE_S:g}s") from exc
    try:
        html = body.decode(response.charset_encoding or "utf-8", errors="replace")
    except LookupError:  # a charset Python doesn't know
        html = body.decode("utf-8", errors="replace")
    return {"url": str(response.url), "status": response.status_code, "text": _visible_text(html, max_chars)}


# ---- save_note ----------------------------------------------------------------


def _note_id(store: VectorStore, session_id: str | None, title: str) -> str:
    """The doc_id for a note titled `title` among the session's notes. Saving
    under an existing title overwrites that note; a different title that
    slugifies the same ("C notes" vs "C++ notes") gets the next free "-2",
    "-3"... suffix instead."""
    slug = slugify(title)
    n = 1
    while True:
        doc_id = slug if n == 1 else f"{slug}-{n}"
        heading = store.first_line(doc_id, owner=session_id)
        if heading is None or heading.lstrip("#").strip().casefold() == title.casefold():
            return doc_id
        n += 1


def build_registry(provider: LLMProvider, store: VectorStore | None = None) -> ToolRegistry:
    """Wire tools to the provider/store they need. Keeping this a factory (rather
    than module-level globals) is what lets tests inject a FakeProvider."""
    settings = get_settings()
    store = store or get_store()
    registry = ToolRegistry()

    async def search_notes(query: str, top_k: int = settings.retrieval_top_k) -> dict:
        chunks = await retrieve(query, provider, store, k=top_k)
        return {
            "query": query,
            "results": [
                {"doc_id": c.doc_id, "score": c.score, "text": c.text} for c in chunks
            ],
        }

    async def save_note(title: str, content: str) -> dict:
        # A private note, like an upload. The shared notes are the same for
        # every visitor, so a note saved there could be read, and overwritten,
        # by anyone using the app.
        session_id = current_session.get()
        title = " ".join(title.split())  # one line, so it stays the note's "# Title"
        doc_id = _note_id(store, session_id, title)
        text = f"# {title}\n\n{content.strip()}\n"
        return await store_private_note(session_id, doc_id, text, store, provider)

    registry.register(
        Tool(
            "search_notes",
            "Semantic search over the user's personal notes. Use this whenever the "
            "user asks about their notes, plans, lists, recipes, or anything they "
            "may have written down. Returns the most relevant passages with doc_ids.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "top_k": {"type": "integer", "description": "Number of passages (default 4)."},
                },
                "required": ["query"],
            },
            search_notes,
        )
    )
    registry.register(
        Tool(
            "save_note",
            "Save a private note for this user, or overwrite their note with the same "
            "title, and index it so it becomes searchable. Only this user's chat can see "
            f"it, and it lasts until they start a new session or for {UPLOAD_TTL_S // 3600} "
            "hours. Use when the user asks to save, remember, or write down something.",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title for the note."},
                    "content": {"type": "string", "description": "Markdown body of the note."},
                },
                "required": ["title", "content"],
            },
            save_note,
        )
    )
    registry.register(
        Tool(
            "calculator",
            "Evaluate an arithmetic expression exactly (+ - * / ** and parentheses; "
            "'17%' is a percentage, '10 % 3' is modulo). Use for any non-trivial "
            "arithmetic instead of computing in your head.",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '0.17 * 2340' or '(3+4)**2'"}
                },
                "required": ["expression"],
            },
            calculator,
        )
    )
    registry.register(
        Tool(
            "current_datetime",
            "Get the current date, time, and weekday in the user's time zone. Use for "
            "anything involving 'today', 'now', deadlines, or elapsed time.",
            {"type": "object", "properties": {}},
            current_datetime,
        )
    )
    registry.register(
        Tool(
            "fetch_url",
            "Fetch a public web page and return its visible text (truncated). Use when "
            "the user gives a URL or asks about the contents of a specific page.",
            {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
                "required": ["url"],
            },
            fetch_url,
        )
    )
    return registry
