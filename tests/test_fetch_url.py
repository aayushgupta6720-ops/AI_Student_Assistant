"""fetch_url against real local HTTP servers. 127.0.0.1 is private, so most
tests tell the tool to treat it as public; the refusal tests don't."""

import threading
import time
import tracemalloc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import app.tools.builtin as builtin
from app.tools.builtin import MAX_FETCH_BYTES, MAX_FETCH_PDF_BYTES, fetch_url
from tests.test_uploads import _pdf

PAGE = b"<html><head><style>p{color:red}</style><script>var secret=1</script></head><body><p>Hello <b>world</b></p></body></html>"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        port = self.server.server_port
        self.server.seen.append((self.path, self.headers.get("Host")))
        if self.path == "/page":
            self._send(200, PAGE)
        elif self.path == "/big":  # 20 MB, far past the read cap
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            block = b"word " * 100_000
            for _ in range(40):
                self.wfile.write(block)
        elif self.path == "/unclosed":  # took minutes with the old backtracking regex
            self._send(200, b"<script>" * (MAX_FETCH_BYTES // 8))
        elif self.path == "/to-ipv6-loopback":
            self._redirect(f"http://[::1]:{port}/page")
        elif self.path == "/loop":
            self._redirect("/loop")
        elif self.path == "/notes.pdf":
            self._send(200, _pdf("Photosynthesis makes glucose"), "application/pdf")
        elif self.path == "/paper":  # a PDF by its type, no .pdf in the URL
            self._send(200, _pdf("Mitochondria make ATP"), "application/pdf; charset=binary")
        elif self.path == "/download?id=42":  # a PDF sent as a generic download
            self._send(200, _pdf("Krebs cycle"), "application/octet-stream")
        elif self.path == "/syllabus.pdf":  # a login page where the PDF was
            self._send(200, b"<html><body><p>Please sign in</p></body></html>")
        elif self.path == "/data.bin":
            self._send(200, bytes(range(256)), "application/octet-stream")
        elif self.path == "/scan.pdf":
            self._send(200, _pdf(None), "application/pdf")
        elif self.path == "/photo.png":
            self._send(200, b"\x89PNG\r\n\x1a\n" + bytes(100), "image/png")
        elif self.path == "/huge.pdf":  # past the PDF cap
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.end_headers()
            for _ in range(MAX_FETCH_PDF_BYTES // 2**20 + 1):
                self.wfile.write(b"%PDF-" + bytes(2**20 - 5))
        elif self.path == "/slow":
            self.send_response(200)
            self.end_headers()
            time.sleep(2)
            self.wfile.write(b"late")

    def _send(self, status, body, content_type="text/html; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def _server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    srv.seen = []  # (path, Host header) of every request received
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


@pytest.fixture
def http_server(_server):
    _server.seen.clear()
    return _server


@pytest.fixture
def server(http_server):
    return f"http://127.0.0.1:{http_server.server_port}"


@pytest.fixture
def loopback_is_public(monkeypatch):
    monkeypatch.setattr(builtin, "_is_public", lambda ip: ip == "127.0.0.1")


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1/", "http://localhost:8000/notes", "http://10.0.0.5/", "http://169.254.169.254/latest/meta-data/",
     "http://[::1]/", "http://[::ffff:127.0.0.1]/"],
)
async def test_refuses_local_and_private_addresses(url):
    with pytest.raises(ValueError, match="private or local"):
        await fetch_url(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/", "example.com"])
async def test_refuses_anything_but_absolute_http_urls(url):
    with pytest.raises(ValueError, match="only absolute http"):
        await fetch_url(url)


async def test_returns_the_visible_text_without_scripts_or_styles(server, loopback_is_public):
    result = await fetch_url(f"{server}/page")
    assert result == {"url": f"{server}/page", "status": 200, "type": "page", "text": "Hello world"}


async def test_checks_every_redirect_hop(server, loopback_is_public):
    with pytest.raises(ValueError, match="::1 is a private or local address"):
        await fetch_url(f"{server}/to-ipv6-loopback")
    with pytest.raises(ValueError, match="more than 5 redirects"):
        await fetch_url(f"{server}/loop")


async def test_connects_to_the_address_it_checked(http_server, loopback_is_public, monkeypatch):
    # httpx used to look the host up again to connect, so a rebinding DNS
    # server could answer "public" to the check and "private" to the
    # connection, and the GET reached the private address. Here the host only
    # resolves through the checked lookup: a second lookup would fail.
    lookups = []

    async def resolve(host, port):
        lookups.append(host)
        return ["127.0.0.1"]

    monkeypatch.setattr(builtin, "_resolve", resolve)
    port = http_server.server_port

    result = await fetch_url(f"http://rebind.invalid:{port}/page")

    assert result["text"] == "Hello world" and result["url"] == f"http://rebind.invalid:{port}/page"
    assert lookups == ["rebind.invalid"]
    assert http_server.seen == [("/page", f"rebind.invalid:{port}")]  # the real name, for virtual hosts


async def test_a_private_answer_is_refused_before_any_request(http_server, monkeypatch):
    async def resolve(host, port):
        return ["127.0.0.1"]

    monkeypatch.setattr(builtin, "_resolve", resolve)

    with pytest.raises(ValueError, match="private or local"):
        await fetch_url(f"http://rebind.invalid:{http_server.server_port}/admin/delete-cache?confirm=1")
    assert http_server.seen == []


async def test_stops_reading_a_huge_page_at_the_cap(server, loopback_is_public):
    tracemalloc.start()
    try:
        result = await fetch_url(f"{server}/big")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["text"].startswith("word word") and len(result["text"]) == 4000
    assert peak < 5 * MAX_FETCH_BYTES  # was several times the 20 MB body


async def test_malformed_html_is_handled_in_linear_time(server, loopback_is_public):
    started = time.perf_counter()
    assert (await fetch_url(f"{server}/unclosed"))["text"] == ""
    assert time.perf_counter() - started < 2


async def test_gives_up_on_a_slow_page_at_the_deadline(server, loopback_is_public, monkeypatch):
    monkeypatch.setattr(builtin, "FETCH_DEADLINE_S", 0.5)
    with pytest.raises(TimeoutError, match="gave up after 0.5s"):
        await fetch_url(f"{server}/slow")


@pytest.mark.parametrize(("path", "text"), [("/notes.pdf", "Photosynthesis makes glucose"), ("/paper", "Mitochondria make ATP")])
async def test_reads_the_text_of_a_pdf_link(server, loopback_is_public, path, text):
    # It used to decode the PDF's bytes as if they were a web page: garbage.
    result = await fetch_url(f"{server}{path}")
    assert result["type"] == "pdf" and text in result["text"]


@pytest.mark.parametrize(
    ("path", "message"),
    [("/scan.pdf", "No text found in that PDF"), ("/huge.pdf", "over 10 MB"),
     ("/photo.png", "that link is image/png, not a web page, text or PDF")],
)
async def test_says_why_it_cant_read_a_file(server, loopback_is_public, path, message):
    with pytest.raises(ValueError, match=message):
        await fetch_url(f"{server}{path}")


async def test_a_pdf_is_recognised_by_its_content(server, loopback_is_public):
    # A PDF sent as a generic download was refused, and a login page at a
    # .pdf address was "a PDF that couldn't be read".
    pdf = await fetch_url(f"{server}/download?id=42")
    page = await fetch_url(f"{server}/syllabus.pdf")

    assert pdf["type"] == "pdf" and "Krebs cycle" in pdf["text"]
    assert page["type"] == "page" and page["text"] == "Please sign in"
    with pytest.raises(ValueError, match="isn't a web page, text or a readable PDF"):
        await fetch_url(f"{server}/data.bin")


# ---- the tool opens only links the user typed --------------------------------


def test_finds_the_links_a_message_contains():
    text = "Read https://Example.com/notes/, www.site.org/syllabus. and (https://a.io/x?id=7)"
    assert builtin.links_in(text) == {"example.com/notes", "site.org/syllabus", "a.io/x?id=7"}


@pytest.mark.parametrize(
    "url, same",
    [
        ("https://www.example.com/notes/", True),  # host case, "www." and a trailing slash aside
        ("http://example.com/notes", True),
        ("https://example.com/notes?d=my+private+note", False),  # a query is where data would go
        ("https://example.com/notes/extra", False),
        ("https://evil.example/notes", False),
    ],
)
def test_a_link_matches_only_the_one_typed(url, same):
    assert (builtin._link_key(url) in builtin.links_in("see example.com/notes")) is same


async def test_the_tool_refuses_a_link_the_user_never_gave(server, loopback_is_public, http_server):
    builtin.user_links.set(builtin.links_in(f"summarise {server}/page please."))

    assert (await builtin.fetch_user_link(f"{server}/page"))["text"] == "Hello world"
    with pytest.raises(PermissionError, match="only links the user typed"):
        await builtin.fetch_user_link(f"{server}/page?d=secret")
    assert [path for path, _ in http_server.seen] == ["/page"]  # the refused one never left
