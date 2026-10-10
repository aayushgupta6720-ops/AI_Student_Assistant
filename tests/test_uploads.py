import io
import json
import time
import tracemalloc
import zlib

import httpx
import pytest

import app.knowledge.uploads as uploads
import app.main as main
from app.api.ratelimit import build_rate_limiters
from app.config import get_settings
from app.intelligence.agent import Agent
from app.intelligence.memory import SessionStore
from app.knowledge.ingest import ingest_dir
from app.knowledge.retrieval import current_session, retrieve
from app.knowledge.store import VectorStore
from app.knowledge.uploads import (
    MAX_SCANNED_PAGES,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_CHARS,
    MAX_UPLOADS_PER_SESSION,
    UPLOAD_TTL_S,
    UploadError,
    as_note,
    extract_text,
    ingest_upload,
)
from app.tools.builtin import build_registry
from tests.fake_provider import FakeProvider, text_turn, tool_turn


def _pdf(text: str | None) -> bytes:
    """A minimal one-page PDF; text=None gives a page with no text layer,
    like a scanned document."""
    content = b"" if text is None else b"BT /F1 12 Tf 72 720 Td (" + text.encode() + b") Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return out


# ---- extraction ------------------------------------------------------------------


def test_extracts_markdown_text_and_pdf():
    assert extract_text("a.md", b"# Cells\n\nMitochondria.\n") == "# Cells\n\nMitochondria."
    assert extract_text("a.TXT", "Café notes".encode()) == "Café notes"
    assert "Photosynthesis makes glucose" in extract_text("bio.pdf", _pdf("Photosynthesis makes glucose"))


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("slides.pptx", b"x", "Only .md, .txt, .pdf"),
        ("big.md", b"x" * (MAX_UPLOAD_BYTES + 1), "over the 2 MB limit"),
        ("latin1.txt", "café".encode("latin-1"), "isn't UTF-8"),
        ("blank.md", b"  \n\n ", "empty"),
        ("scan.pdf", _pdf(None), "Scanned PDFs"),
        ("broken.pdf", b"%PDF-1.4 not really", "couldn't be read"),
        # pypdf raises AttributeError here, not PdfReadError: it used to be a 500
        ("no-key.pdf", _pdf("x").replace(b"/Root", b"/Encrypt 99 0 R /Root"), "couldn't be read"),
        ("long.md", b"x" * (MAX_UPLOAD_CHARS + 1), "over 200,000 characters"),
    ],
)
def test_refuses_what_it_cannot_index_with_a_readable_reason(filename, data, message):
    with pytest.raises(UploadError, match=message):
        extract_text(filename, data)


def test_a_small_pdf_with_too_much_text_is_refused(monkeypatch):
    # The byte cap doesn't bound a compressed PDF's text; the character cap does.
    monkeypatch.setattr(uploads, "MAX_UPLOAD_CHARS", 10)
    with pytest.raises(UploadError, match="over 10 characters"):
        extract_text("dense.pdf", _pdf("far more than ten characters"))


@pytest.mark.parametrize("encode", [
    lambda text: text.replace("\n", "\r\n").encode(),  # saved on Windows
    lambda text: b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode(),  # ...with a byte-order mark
])
def test_windows_text_files_read_like_any_other(encode):
    note = "# Biology lectures\n\n## Week 1\n\nCells.\n\n## Week 2\n\nMitosis."

    assert extract_text("bio.md", encode(note)) == note
    assert as_note("bio.md", extract_text("bio.md", encode(note)))[1].startswith("# Biology lectures\n")


def test_notes_without_a_title_get_one_from_the_file_name():
    assert as_note("Lecture_3-cells.txt", "Mitochondria.") == ("lecture-3-cells", "# Lecture 3 cells\n\nMitochondria.")
    assert as_note("x.md", "# Own title\n\nBody") == ("x", "# Own title\n\nBody")


def _pdf_pages(streams: list[bytes], compress: bool = True) -> bytes:
    """A PDF with one page per content stream (raw drawing instructions)."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for ops in streams:
        body = zlib.compress(ops, 9) if compress else ops
        objects.append(b"<< /Length %d%s >>\nstream\n" % (len(body), b" /Filter /FlateDecode" if compress else b"")
                       + body + b"\nendstream")
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % len(objects))
        kids.append(len(objects))
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)


def _text(words: str) -> bytes:
    return b"BT /F1 12 Tf 72 720 Td (" + words.encode() + b") Tj ET"


def _drawing(size: int) -> bytes:
    return b"0 0 m\n" * (size // 6)


@pytest.mark.parametrize(
    "streams",
    [
        [_drawing(8 * 1024 * 1024)],  # 13 KB compressed; took memory from 66 MB to 496 MB
        [_drawing(uploads.MAX_PDF_PAGE_BYTES + 6)] * 3,  # just past the per-page limit
    ],
)
def test_a_pdf_of_huge_drawings_is_refused_cheaply(streams):
    data = _pdf_pages(streams)
    tracemalloc.start()
    started = time.perf_counter()
    try:
        with pytest.raises(UploadError, match="too complex to read"):
            extract_text("bomb.pdf", data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(data) < 50_000
    assert peak < 20 * 1024 * 1024 and time.perf_counter() - started < 1


def test_text_pages_are_read_around_a_skipped_drawing():
    data = _pdf_pages([_text("Mitosis has four phases"), _drawing(2 * 1024 * 1024), _text("Meiosis halves chromosomes")])

    text = extract_text("biology.pdf", data)

    assert "Mitosis has four phases" in text and "Meiosis halves chromosomes" in text


def test_a_pdf_with_too_many_detailed_pages_is_refused(monkeypatch):
    # Each page is under the per-page limit, but parsing them all would hold
    # the CPU (and the GIL) for as long as the pages keep coming.
    monkeypatch.setattr(uploads, "MAX_PDF_TOTAL_BYTES", 3000)
    data = _pdf_pages([_drawing(1000) + _text("page")] * 5)

    with pytest.raises(UploadError, match="too many detailed pages"):
        extract_text("slides.pdf", data)


# ---- storing uploads ---------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "s.sqlite")
    s.upsert_doc("shared-note", ["shared note about travel"], [[1.0, 0, 0, 0]])
    return s


async def test_an_upload_is_visible_to_its_session_only(store):
    result = await ingest_upload("alice", "cells.md", b"Mitochondria make ATP.", store, FakeProvider(turns=[]))

    assert result == {"doc_id": "cells", "chunks": 1}
    assert [d["doc_id"] for d in store.list_docs(owner="alice")] == ["shared-note", "cells"]
    assert [d["doc_id"] for d in store.list_docs(owner="bob")] == ["shared-note"]
    [chunk] = [h for h in store.search([1, 1, 1, 1], k=5, owner="alice") if h.doc_id == "cells"]
    assert chunk.text.startswith("# cells\n\n")  # labelled, so search results name the note


async def test_upload_limit_counts_distinct_notes_and_replacing_one_is_allowed(store):
    provider = FakeProvider(turns=[])
    for i in range(MAX_UPLOADS_PER_SESSION):
        await ingest_upload("alice", f"n{i}.md", b"text", store, provider)

    await ingest_upload("alice", "n0.md", b"new version", store, provider)  # same name: replaces
    with pytest.raises(UploadError, match="up to 10 private notes"):
        await ingest_upload("alice", "one-more.md", b"text", store, provider)
    await ingest_upload("bob", "one-more.md", b"text", store, provider)  # other sessions unaffected


async def test_reingesting_shared_notes_never_prunes_uploads(tmp_path, store):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "shared-note.md").write_text("shared note")
    await ingest_upload("alice", "cells.md", b"Mitochondria.", store, FakeProvider(turns=[]))

    await ingest_dir(notes, store, FakeProvider(turns=[]))

    assert [d["doc_id"] for d in store.list_docs(owner="alice")] == ["shared-note", "cells"]


# ---- session scoping of search ---------------------------------------------------------


@pytest.fixture
def every_match(monkeypatch):
    """Keep weak matches too: these tests are about which notes a session can
    see, and the fake embeddings score the shared note far below the upload."""
    monkeypatch.setattr(get_settings(), "retrieval_score_margin", 2.0)  # cosine scores span 2


async def test_retrieve_only_sees_the_current_sessions_uploads(store, every_match):
    provider = FakeProvider(turns=[])
    await ingest_upload("alice", "cells.md", b"Mitochondria.", store, provider)

    current_session.set("bob")
    assert {c.doc_id for c in await retrieve("mitochondria", provider, store, k=5)} == {"shared-note"}
    current_session.set("alice")
    assert {c.doc_id for c in await retrieve("mitochondria", provider, store, k=5)} == {"shared-note", "cells"}


async def test_agent_searches_are_scoped_to_the_session_it_serves(store, every_match):
    provider = FakeProvider([tool_turn("search_notes", {"query": "mitochondria"}), text_turn("ok")] * 2)
    await ingest_upload("alice", "cells.md", b"Mitochondria.", store, provider)
    agent = Agent(provider, build_registry(provider, store), SessionStore())

    async def found(session):
        events = [e async for e in agent.run_turn(session, "what are mitochondria?")]
        [result] = [e for e in events if type(e).__name__ == "AgentToolResult"]
        return {r["doc_id"] for r in result.result["results"]}

    assert await found("alice") == {"shared-note", "cells"}
    assert await found("bob") == {"shared-note"}


async def test_the_agent_can_read_a_whole_upload_and_lists_it_as_a_source(store):
    # For a question that needs all of a note, the model reads it with read_note;
    # the note then shows as a source of the answer, as it would after a search.
    provider = FakeProvider([tool_turn("read_note", {"doc_id": "cells"}), text_turn("ok")])
    await ingest_upload("alice", "cells.md", b"Mitochondria make ATP.\n\nRibosomes make proteins.", store, provider)
    agent = Agent(provider, build_registry(provider, store), SessionStore())

    events = [e async for e in agent.run_turn("alice", "summarise my cells note")]

    [result] = [e for e in events if type(e).__name__ == "AgentToolResult"]
    assert result.result["text"] == "# cells\n\nMitochondria make ATP.\n\nRibosomes make proteins."  # as uploaded
    assert events[-1].sources == ["cells"]


async def test_expired_uploads_stop_showing_up_without_waiting_for_another_upload(store, monkeypatch):
    provider = FakeProvider(turns=[])
    await ingest_upload("alice", "cells.md", b"Mitochondria.", store, provider)
    later = time.time() + UPLOAD_TTL_S + 60
    monkeypatch.setattr(time, "time", lambda: later)

    current_session.set("alice")
    assert {c.doc_id for c in await retrieve("mitochondria", provider, store, k=5)} == {"shared-note"}


# ---- the HTTP API ------------------------------------------------------------------


@pytest.fixture
async def visitors(store, monkeypatch):
    """A client per visitor, each with its own cookie jar as a browser has: the
    server issues each one a session the first time it calls."""
    state = {"store": store, "provider": FakeProvider(turns=[]), "memory": SessionStore(),
             "rate_limiters": build_rate_limiters(get_settings())}
    for name, value in state.items():
        monkeypatch.setattr(main.app.state, name, value, raising=False)
    clients: dict[str, httpx.AsyncClient] = {}

    def visitor(name: str) -> httpx.AsyncClient:
        if name not in clients:
            clients[name] = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")
        return clients[name]

    yield visitor
    for client in clients.values():
        await client.aclose()


@pytest.fixture
async def api(visitors):
    return visitors("alice")


async def test_upload_list_delete_and_reset_through_the_api(visitors):
    alice, bob = visitors("alice"), visitors("bob")
    r = await alice.post("/notes/upload", files={"file": ("cells.md", b"Mitochondria.")})
    assert r.status_code == 200 and r.json() == {"doc_id": "cells", "chunks": 1}
    await alice.post("/notes/upload", files={"file": ("dna.pdf", _pdf("DNA is a double helix"))})

    docs = (await alice.get("/notes")).json()["docs"]
    assert [(d["doc_id"], d["uploaded"]) for d in docs] == [("shared-note", False), ("cells", True), ("dna", True)]
    assert [d["doc_id"] for d in (await bob.get("/notes")).json()["docs"]] == ["shared-note"]
    assert (await bob.delete("/notes/cells")).json() == {"deleted": False}  # not bob's to delete

    assert (await alice.delete("/notes/shared-note")).json() == {"deleted": False}
    assert (await alice.delete("/notes/cells")).json() == {"deleted": True}

    await alice.post("/reset", params={"keep_uploads": "true"})  # page reload: uploads stay
    assert len((await alice.get("/notes")).json()["docs"]) == 2
    await alice.post("/reset")  # New session: uploads go
    assert len((await alice.get("/notes")).json()["docs"]) == 1


async def test_a_session_is_issued_by_the_server_not_chosen_by_the_client(visitors):
    alice = visitors("alice")
    first = await alice.get("/notes")
    cookie = first.headers["set-cookie"]
    assert cookie.startswith("sid=") and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert "set-cookie" not in (await alice.get("/notes")).headers  # kept, not reissued

    # An id the client makes up (the README's curl example once used "cli") is replaced.
    chosen = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test",
                               cookies={"sid": "cli"})
    async with chosen:
        r = await chosen.get("/notes")
    issued = r.headers["set-cookie"].split(";")[0].removeprefix("sid=")
    assert issued != "cli" and len(issued) == 32


async def test_the_notes_list_drops_expired_uploads(api, monkeypatch):
    await api.post("/notes/upload", files={"file": ("cells.md", b"Mitochondria.")})
    later = time.time() + UPLOAD_TTL_S + 60
    monkeypatch.setattr(time, "time", lambda: later)

    docs = (await api.get("/notes")).json()["docs"]
    assert [d["doc_id"] for d in docs] == ["shared-note"]


TWO_CHUNKS = (("a " * 300).strip() + "\n\n" + ("b " * 300).strip()).encode()


async def _upload(client: httpx.AsyncClient, name: str, data: bytes) -> httpx.Response:
    return await client.post("/notes/upload", files={"file": (name, data)})


async def test_uploads_count_their_chunks_against_a_daily_budget(visitors, monkeypatch):
    # Counting files alone let one visitor, switching sessions, upload ~500
    # chunks at a time all day: more memory than a free instance has. The
    # budget is per visitor (IP), so a fresh session doesn't reset it.
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_chunk_limit_per_day", 3)
    monkeypatch.setattr(main.app.state, "rate_limiters", build_rate_limiters(settings))

    assert (await _upload(visitors("first tab"), "big.md", TWO_CHUNKS)).json() == {"doc_id": "big", "chunks": 2}
    fresh = visitors("new session, same visitor")
    refused = await _upload(fresh, "big2.md", TWO_CHUNKS)

    assert refused.status_code == 429 and int(refused.headers["Retry-After"]) == 24 * 3600
    assert refused.json()["detail"] == (
        "That's 2 note chunks, which would take you over the limit of 3 note chunks a day. Try again in 24 hours."
    )
    assert [d["doc_id"] for d in (await fresh.get("/notes")).json()["docs"]] == [
        "shared-note"
    ]  # refused before anything was embedded or stored
    assert (await _upload(fresh, "small.md", b"tiny")).status_code == 200  # 1 more still fits


async def test_private_notes_stop_at_the_apps_memory_cap_across_all_sessions(visitors, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_private_chunks", 3)
    assert (await _upload(visitors("alice"), "big.md", TWO_CHUNKS)).status_code == 200
    assert (await _upload(visitors("bob"), "small.md", b"tiny")).status_code == 200

    full = await _upload(visitors("carol"), "small.md", b"tiny")

    assert full.status_code == 503 and "as many private notes as it has room for" in full.json()["detail"]
    assert (await _upload(visitors("alice"), "big.md", b"now one chunk")).status_code == 200  # replacing frees its chunks


async def test_upload_errors_come_back_as_400_with_the_reason(api):
    r = await api.post("/notes/upload", files={"file": ("x.exe", b"MZ")})

    assert r.status_code == 400
    assert "Only .md, .txt, .pdf" in r.json()["detail"]


async def test_notes_saved_from_chat_count_against_the_daily_chunk_budget(visitors, store, monkeypatch):
    # save_note didn't charge the budget, so chat alone could fill the
    # server-wide private-chunk cap and lock everyone's uploads out.
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_chunk_limit_per_day", 3)
    monkeypatch.setattr(main.app.state, "rate_limiters", build_rate_limiters(settings))
    essay = TWO_CHUNKS.decode()
    provider = FakeProvider([
        tool_turn("save_note", {"title": "Essay", "content": essay}), text_turn("Saved."),
        tool_turn("save_note", {"title": "Essay two", "content": essay}), text_turn("Couldn't save it."),
    ])
    monkeypatch.setattr(main.app.state, "agent", Agent(provider, build_registry(provider, store), SessionStore()),
                        raising=False)

    async def saved(session):
        r = await visitors(session).post("/chat", json={"message": "save my essay"})
        [result] = [json.loads(block.split("data: ", 1)[1]) for block in r.text.split("\n\n")
                    if block.startswith("event: tool_result")]
        return result

    assert not (await saved("alice"))["is_error"]  # 2 of the 3 chunks
    refused = await saved("new-session")
    assert refused["is_error"] and "over the limit of 3 note chunks a day" in refused["preview"]
    assert store.private_count() == 2
    assert (await _upload(visitors("bob"), "big.md", TWO_CHUNKS)).status_code == 429  # same visitor, same budget


async def test_uploads_with_non_latin_names_are_kept_apart(store):
    # Every non-Latin name used to become "note", so each upload replaced the last.
    for name in ("Биология.md", "Химия.md", "生物.txt"):
        await ingest_upload("alice", name, b"text", store, FakeProvider(turns=[]))

    assert {d["doc_id"] for d in store.list_docs(owner="alice") if d["uploaded"]} == {"биология", "химия", "生物"}


async def test_a_private_note_never_takes_a_shared_notes_id(store):
    # Sharing an id mixed the two notes up in searches and source chips.
    provider = FakeProvider(turns=[])
    upload = await ingest_upload("alice", "Shared Note.md", b"mine", store, provider)
    current_session.set("bob")
    saved = await build_registry(provider, store).execute("save_note", {"title": "Shared note", "content": "mine"})

    assert upload["doc_id"] == "shared-note-2" and saved.result["doc_id"] == "shared-note-2"
    assert store.first_line("shared-note") is not None  # the shared note itself untouched


# ---- scanned PDFs ------------------------------------------------------------------


def _scan(pages: int = 1) -> bytes:
    """A PDF of `pages` pages with no text layer, like a scanned document."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(612, 792)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


async def test_a_scanned_pdf_is_transcribed_by_the_model_and_stored_like_any_note(store):
    provider = FakeProvider(turns=[], transcription="Enzymes lower activation energy.")
    charged = []

    result = await ingest_upload("alice", "lecture-7.pdf", _scan(2), store, provider, charge_transcription=lambda: charged.append(1))

    assert result == {"doc_id": "lecture-7", "chunks": 1, "transcribed": True}
    assert provider.transcribed == [_scan(2)] and charged == [1]
    assert store.note_text("lecture-7", owner="alice") == "# lecture 7\n\nEnzymes lower activation energy."


async def test_a_pdf_with_text_is_never_sent_to_the_model(store):
    provider = FakeProvider(turns=[], transcription="should not be used")

    result = await ingest_upload("alice", "dna.pdf", _pdf("DNA is a double helix"), store, provider,
                                 charge_transcription=lambda: pytest.fail("charged for a PDF with text"))

    assert provider.transcribed == [] and "transcribed" not in result


async def test_a_long_scan_is_refused_before_any_model_call(store):
    provider = FakeProvider(turns=[], transcription="text")

    with pytest.raises(UploadError, match=f"up to {MAX_SCANNED_PAGES} pages; this one has {MAX_SCANNED_PAGES + 1}"):
        await ingest_upload("alice", "book.pdf", _scan(MAX_SCANNED_PAGES + 1), store, provider)
    assert provider.transcribed == []


async def test_a_scan_the_session_has_no_room_for_costs_no_model_call(store):
    provider = FakeProvider(turns=[], transcription="text")
    for i in range(MAX_UPLOADS_PER_SESSION):
        await ingest_upload("alice", f"n{i}.md", b"text", store, provider)

    with pytest.raises(UploadError, match="up to 10 private notes"):
        await ingest_upload("alice", "scan.pdf", _scan(), store, provider,
                            charge_transcription=lambda: pytest.fail("charged for a scan that can't be stored"))
    assert provider.transcribed == []


@pytest.mark.parametrize(("transcription", "finish", "message"), [
    ("", "stop", "No text could be read"),
    ("Chapter 1 of a novel", "recitation", "published text"),
    ("half of it", "max_tokens", "Split it into smaller files"),
    ("blocked", "safety", "safety filters"),
    ("x" * (MAX_UPLOAD_CHARS + 1), "stop", "over 200,000 characters"),
])
async def test_a_transcription_that_did_not_work_is_refused_with_the_reason(store, transcription, finish, message):
    provider = FakeProvider(turns=[], transcription=transcription, transcribe_finish=finish)

    with pytest.raises(UploadError, match=message):
        await ingest_upload("alice", "scan.pdf", _scan(), store, provider)
    assert [d["doc_id"] for d in store.list_docs(owner="alice")] == ["shared-note"]  # nothing half-stored


async def test_scanned_pdfs_have_their_own_daily_limit(api, monkeypatch):
    # Each one is a chat-model request from the quota every visitor's chats share.
    settings = get_settings()
    monkeypatch.setattr(settings, "transcribe_limit_per_day", 1)
    monkeypatch.setattr(main.app.state, "rate_limiters", build_rate_limiters(settings))
    main.app.state.provider.transcription = "Scanned notes."

    first = await _upload(api, "scan1.pdf", _scan())
    second = await _upload(api, "scan2.pdf", _scan())

    assert first.json() == {"doc_id": "scan1", "chunks": 1, "transcribed": True}
    assert second.status_code == 429 and "1 scanned PDFs a day" in second.json()["detail"]
    assert (await _upload(api, "typed.md", b"Typed notes.")).status_code == 200  # other uploads unaffected


async def test_an_uploads_log_line_says_how_long_each_step_took(api, monkeypatch):
    # The first live scanned upload took 13 s against 3 s locally, and its log
    # couldn't say whether transcribing or embedding was the slow part.
    import app.api.routes as routes

    logged = []
    monkeypatch.setattr(routes, "log_event", lambda **fields: logged.append(fields))
    monkeypatch.setattr(get_settings(), "log_chat_text", False)
    main.app.state.provider.transcription = "Scanned notes."

    await _upload(api, "bank-pin-4321.pdf", _scan())

    [line] = [f for f in logged if f["event"] == "note_uploaded"]
    assert [s["name"] for s in line["steps"]] == ["extract_text", "transcribe_pdf", "embed_documents", "store_upsert"]
    assert line["latency_ms"] > 0 and set(line["per_layer_ms"]) == {"knowledge", "inference"}
    assert "bank-pin-4321" not in json.dumps(line)  # a private note's name is its file name
    assert line["steps"][2]["meta"] == {"doc_id": "<13 chars>", "chunks": 1}


async def test_a_failed_uploads_log_line_has_its_timings_too(api, monkeypatch):
    import app.api.routes as routes
    from app.inference.provider import ModelTimeoutError

    logged = []
    monkeypatch.setattr(routes, "log_event", lambda **fields: logged.append(fields))

    async def stalled(data):
        raise ModelTimeoutError("no response from Gemini within 60s")

    monkeypatch.setattr(main.app.state.provider, "transcribe_pdf", stalled)

    r = await _upload(api, "scan.pdf", _scan())

    [line] = [f for f in logged if f["event"] == "upload_failed"]
    assert r.status_code == 503 and [s["name"] for s in line["steps"]] == ["extract_text", "transcribe_pdf"]
