"""fetch_url against real local HTTP servers. 127.0.0.1 is private, so most
tests tell the tool to treat it as public; the refusal tests don't."""

import threading
import time
import tracemalloc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import app.tools.builtin as builtin
from app.tools.builtin import MAX_FETCH_BYTES, fetch_url

PAGE = b"<html><head><style>p{color:red}</style><script>var secret=1</script></head><body><p>Hello <b>world</b></p></body></html>"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        port = self.server.server_port
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
        elif self.path == "/slow":
            self.send_response(200)
            self.end_headers()
            time.sleep(2)
            self.wfile.write(b"late")

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


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
    assert result == {"url": f"{server}/page", "status": 200, "text": "Hello world"}


async def test_checks_every_redirect_hop(server, loopback_is_public):
    with pytest.raises(ValueError, match="::1 is a private or local address"):
        await fetch_url(f"{server}/to-ipv6-loopback")
    with pytest.raises(ValueError, match="more than 5 redirects"):
        await fetch_url(f"{server}/loop")


async def test_checks_the_address_it_actually_connected_to(server, monkeypatch):
    # DNS said public but the connection landed somewhere private (rebinding)
    async def dns_says_public(url):
        pass

    monkeypatch.setattr(builtin, "_check_public_url", dns_says_public)
    with pytest.raises(ValueError, match="private or local"):
        await fetch_url(f"{server}/page")


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
