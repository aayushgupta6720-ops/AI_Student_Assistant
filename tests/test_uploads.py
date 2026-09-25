import time

import httpx
import pytest

import app.knowledge.uploads as uploads
import app.main as main
from app.intelligence.agent import Agent
from app.intelligence.memory import SessionStore
from app.knowledge.ingest import ingest_dir
from app.knowledge.retrieval import current_session, retrieve
from app.knowledge.store import VectorStore
from app.knowledge.uploads import (
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


def test_notes_without_a_title_get_one_from_the_file_name():
    assert as_note("Lecture_3-cells.txt", "Mitochondria.") == ("lecture-3-cells", "# Lecture 3 cells\n\nMitochondria.")
    assert as_note("x.md", "# Own title\n\nBody") == ("x", "# Own title\n\nBody")


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


async def test_retrieve_only_sees_the_current_sessions_uploads(store):
    provider = FakeProvider(turns=[])
    await ingest_upload("alice", "cells.md", b"Mitochondria.", store, provider)

    current_session.set("bob")
    assert {c.doc_id for c in await retrieve("mitochondria", provider, store, k=5)} == {"shared-note"}
    current_session.set("alice")
    assert {c.doc_id for c in await retrieve("mitochondria", provider, store, k=5)} == {"shared-note", "cells"}


async def test_agent_searches_are_scoped_to_the_session_it_serves(store):
    provider = FakeProvider([tool_turn("search_notes", {"query": "mitochondria"}), text_turn("ok")] * 2)
    await ingest_upload("alice", "cells.md", b"Mitochondria.", store, provider)
    agent = Agent(provider, build_registry(provider, store), SessionStore())

    async def found(session):
        events = [e async for e in agent.run_turn(session, "what are mitochondria?")]
        [result] = [e for e in events if type(e).__name__ == "AgentToolResult"]
        return {r["doc_id"] for r in result.result["results"]}

    assert await found("alice") == {"shared-note", "cells"}
    assert await found("bob") == {"shared-note"}


async def test_expired_uploads_stop_showing_up_without_waiting_for_another_upload(store, monkeypatch):
    provider = FakeProvider(turns=[])
    await ingest_upload("alice", "cells.md", b"Mitochondria.", store, provider)
    later = time.time() + UPLOAD_TTL_S + 60
    monkeypatch.setattr(time, "time", lambda: later)

    current_session.set("alice")
    assert {c.doc_id for c in await retrieve("mitochondria", provider, store, k=5)} == {"shared-note"}


# ---- the HTTP API ------------------------------------------------------------------


@pytest.fixture
async def api(store, monkeypatch):
    for name, value in {"store": store, "provider": FakeProvider(turns=[]), "memory": SessionStore()}.items():
        monkeypatch.setattr(main.app.state, name, value, raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        yield client


async def test_upload_list_delete_and_reset_through_the_api(api):
    r = await api.post("/notes/upload", data={"session_id": "alice"}, files={"file": ("cells.md", b"Mitochondria.")})
    assert r.status_code == 200 and r.json() == {"doc_id": "cells", "chunks": 1}
    await api.post("/notes/upload", data={"session_id": "alice"}, files={"file": ("dna.pdf", _pdf("DNA is a double helix"))})

    alice = (await api.get("/notes", params={"session_id": "alice"})).json()["docs"]
    assert [(d["doc_id"], d["uploaded"]) for d in alice] == [("shared-note", False), ("cells", True), ("dna", True)]
    bob = (await api.get("/notes", params={"session_id": "bob"})).json()["docs"]
    assert [d["doc_id"] for d in bob] == ["shared-note"]

    assert (await api.delete("/notes/shared-note", params={"session_id": "alice"})).json() == {"deleted": False}
    assert (await api.delete("/notes/cells", params={"session_id": "alice"})).json() == {"deleted": True}

    await api.post("/reset/alice", params={"keep_uploads": "true"})  # page reload: uploads stay
    assert len((await api.get("/notes", params={"session_id": "alice"})).json()["docs"]) == 2
    await api.post("/reset/alice")  # New session: uploads go
    assert len((await api.get("/notes", params={"session_id": "alice"})).json()["docs"]) == 1


async def test_the_notes_list_drops_expired_uploads(api, monkeypatch):
    await api.post("/notes/upload", data={"session_id": "alice"}, files={"file": ("cells.md", b"Mitochondria.")})
    later = time.time() + UPLOAD_TTL_S + 60
    monkeypatch.setattr(time, "time", lambda: later)

    docs = (await api.get("/notes", params={"session_id": "alice"})).json()["docs"]
    assert [d["doc_id"] for d in docs] == ["shared-note"]


async def test_upload_errors_come_back_as_400_with_the_reason(api):
    r = await api.post("/notes/upload", data={"session_id": "alice"}, files={"file": ("x.exe", b"MZ")})

    assert r.status_code == 400
    assert "Only .md, .txt, .pdf" in r.json()["detail"]
