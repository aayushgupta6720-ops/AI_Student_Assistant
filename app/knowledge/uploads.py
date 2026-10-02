"""Private notes: uploaded files, and notes saved from the chat with
save_note. Each is chunked, embedded and stored under its session, where only
that session's searches can see it (see store.py). They expire with the
session, or after UPLOAD_TTL_S, whichever comes first."""

import asyncio
import contextvars
import io
from pathlib import Path
from typing import Callable

from pypdf import PageObject, PdfReader, apply_configuration
from pypdf.errors import LimitReachedError

from app.config import get_settings
from app.inference.provider import LLMProvider
from app.knowledge.ingest import chunk_note, embedded_with, slugify
from app.knowledge.store import VectorStore
from app.observability import time_step

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
# The byte cap alone doesn't bound the work: a compressed PDF well under it
# can hold megabytes of text, and every ~800 chars is another chunk to embed.
# 200,000 chars is about 100 pages of notes, and a few embed requests.
MAX_UPLOAD_CHARS = 200_000
MAX_UPLOADS_PER_SESSION = 10
UPLOAD_TTL_S = 24 * 3600
# pypdf holds ~55 bytes of memory per byte of a page's drawing instructions
# while it extracts text, parses them at ~2 MB/s in pure Python (holding the
# GIL, so every other request slows), and by default expands a compressed
# stream to 75 MB: a 13 KB PDF took the process from 66 MB to a 496 MB peak,
# on a free instance with 512 MB. So no stream expands past 2 MB, a page with
# over 512 KB of instructions (a detailed drawing; a page of text is tens of
# KB) is skipped, and a PDF gets at most 4 MB of instructions parsed in all.
MAX_PDF_STREAM_BYTES = 2 * 1024 * 1024
MAX_PDF_PAGE_BYTES = 512 * 1024
MAX_PDF_TOTAL_BYTES = 4 * 1024 * 1024
_PDF_LIMITS = dict.fromkeys(
    ["zlib_maximum_output_length", "lzw_maximum_output_length", "run_length_maximum_output_length",
     "array_based_stream_maximum_output_length"],
    MAX_PDF_STREAM_BYTES,
)
ALLOWED_SUFFIXES = (".md", ".txt", ".pdf")
# A PDF with no text layer (a scan) is read by the model instead: one request
# from the shared daily quota, and it gets slower with every page.
MAX_SCANNED_PAGES = 10
_TRANSCRIBE_REFUSALS = {
    "max_tokens": "That scanned PDF has more text than can be transcribed in one go. Split it into smaller files.",
    "recitation": "Gemini wouldn't transcribe that PDF: it looked like published text it can't reproduce.",
    "safety": "Gemini's safety filters blocked the transcription of that PDF.",
}


# The API's per-visitor chunk budget, for notes save_note stores: that runs
# inside the agent, far from the request that knows who the visitor is. The
# chat route sets it each turn; unset, nothing is charged.
note_charge: contextvars.ContextVar[Callable[[int], None] | None] = contextvars.ContextVar(
    "note_charge", default=None
)


class UploadError(ValueError):
    """An upload we refuse, with a message fit to show the user."""


class NoTextLayerError(UploadError):
    """A PDF whose pages have no text to extract, as in a scan."""

    def __init__(self, pages: int) -> None:
        super().__init__("No text found in that PDF. Scanned PDFs without a text layer aren't supported.")
        self.pages = pages


class PrivateNotesFullError(UploadError):
    """The app holds as many private notes as it has memory for. Nothing is
    wrong with the upload; it can work once older notes expire."""


def pdf_text(data: bytes, max_chars: int) -> str:
    """A PDF's text, page by page, stopping once there's more than max_chars
    of it: later pages aren't extracted at all. Used for uploads and for PDF
    links fetch_url reads. Raises UploadError if there's no text to get:
    NoTextLayerError if the pages simply have none."""
    pages: list[str] = []
    length = skipped = parsed = page_count = 0
    try:
        with apply_configuration(**_PDF_LIMITS):
            reader = PdfReader(io.BytesIO(data))
            page_count = len(reader.pages)
            for page in reader.pages:
                size = _content_size(page)
                if size is None or size > MAX_PDF_PAGE_BYTES:
                    skipped += 1
                    continue
                parsed += size
                if parsed > MAX_PDF_TOTAL_BYTES:
                    raise UploadError("That PDF has too many detailed pages to read. Split it into smaller files.")
                pages.append((page.extract_text() or "").strip())
                length += len(pages[-1])
                if length > max_chars:
                    break  # already enough; don't extract the rest
    except UploadError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf raises all sorts on a broken file
        # Not just pypdf's PdfReadError: a PDF naming an /Encrypt object it
        # doesn't have raised AttributeError, which came back as a 500.
        raise UploadError("That PDF couldn't be read.") from exc
    text = "\n\n".join(pages)
    if not text.strip():
        if skipped:
            raise UploadError("That PDF's pages are too complex to read: they're drawings rather than text.")
        raise NoTextLayerError(page_count)
    return text


def _content_size(page: PageObject) -> int | None:
    """The size of a page's drawing instructions, measured before extract_text
    parses them (the costly part); None if they expand past the stream limit."""
    try:
        contents = page.get_contents()  # expands the stream, within _PDF_LIMITS
    except LimitReachedError:
        return None
    return 0 if contents is None else len(contents.get_data())


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadError(f"Only {', '.join(ALLOWED_SUFFIXES)} files can be uploaded.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"That file is over the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    if suffix == ".pdf":
        text = pdf_text(data, MAX_UPLOAD_CHARS)
    else:
        try:
            # utf-8-sig drops a byte-order mark, which hid a note's own "# Title".
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UploadError("That file isn't UTF-8 text.") from exc
        # Windows line endings: the chunker splits paragraphs on "\n\n", so a
        # CRLF file was one long paragraph, cut every 700 chars mid-word.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise UploadError("That file is empty.")

    text = text.strip()
    if len(text) > MAX_UPLOAD_CHARS:
        raise UploadError(f"That file has over {MAX_UPLOAD_CHARS:,} characters of text. Split it into smaller files.")
    return text


def clear_of_shared(store: VectorStore, doc_id: str) -> str:
    """doc_id, or doc_id-2, -3...: the first that no shared note uses. A private
    note named like a shared one ("Reading List.md") would otherwise share its
    doc_id, and searches, source chips and their passages would mix them up."""
    shared = {d["doc_id"] for d in store.list_docs()}
    n, candidate = 1, doc_id
    while candidate in shared:
        n += 1
        candidate = f"{doc_id}-{n}"
    return candidate


def as_note(filename: str, text: str) -> tuple[str, str]:
    """(doc_id, markdown) for an uploaded file. The note gets a "# Title"
    first line if it lacks one, so the heading-aware chunker labels every
    chunk with the note it came from."""
    stem = Path(filename).stem
    if not text.startswith("# "):
        title = stem.replace("_", " ").replace("-", " ").strip() or "Untitled"
        text = f"# {title}\n\n{text}"
    return slugify(stem), text


def _own_notes(store: VectorStore, session_id: str, doc_id: str) -> dict[str, int]:
    """The session's private notes (doc_id -> chunks), after refusing a new
    note `doc_id` if the session already has as many as it may."""
    store.purge_uploads(UPLOAD_TTL_S)
    own = {d["doc_id"]: d["chunks"] for d in store.list_docs(owner=session_id) if d["uploaded"]}
    if doc_id not in own and len(own) >= MAX_UPLOADS_PER_SESSION:
        raise UploadError(
            f"You can have up to {MAX_UPLOADS_PER_SESSION} private notes per chat "
            "(uploads and saved notes); remove one first."
        )
    return own


async def store_private_note(
    session_id: str | None,
    doc_id: str,
    text: str,
    store: VectorStore,
    provider: LLMProvider,
    charge: Callable[[int], None] | None = None,
) -> dict:
    """Chunk, embed and store `text` as one of `session_id`'s private notes.
    Storing a doc_id the session already has replaces that note. `charge` is
    called with the chunk count once the note is accepted, before anything is
    embedded, and may raise to refuse it (the API's per-visitor budget);
    without one, note_charge's is used."""
    if not session_id:
        raise UploadError("Private notes need a session.")

    own = _own_notes(store, session_id, doc_id)
    chunks = chunk_note(text)
    # A visitor can start any number of sessions, so the per-session cap
    # doesn't bound memory; this one does. Replacing a note frees its chunks.
    if store.private_count() - own.get(doc_id, 0) + len(chunks) > get_settings().max_private_chunks:
        raise PrivateNotesFullError(
            "The assistant is holding as many private notes as it has room for. "
            "Try again later, once older notes have expired."
        )
    charge = charge or note_charge.get()
    if charge is not None:
        charge(len(chunks))
    with time_step("inference", "embed_documents", doc_id=doc_id, chunks=len(chunks)):
        embeddings = await provider.embed(chunks, "document")
    with time_step("knowledge", "store_upsert", doc_id=doc_id):
        store.upsert_doc(doc_id, chunks, embeddings, owner=session_id, embedded_with=embedded_with(), source=text)
    return {"doc_id": doc_id, "chunks": len(chunks)}


async def ingest_upload(
    session_id: str,
    filename: str,
    data: bytes,
    store: VectorStore,
    provider: LLMProvider,
    charge: Callable[[int], None] | None = None,
    charge_transcription: Callable[[], None] | None = None,
) -> dict:
    """Store an uploaded file as one of `session_id`'s private notes.
    Re-uploading a file with the same name replaces the earlier version. A
    scanned PDF is transcribed by the model first; `charge_transcription` is
    called just before, and may raise to refuse it (the API's daily limit)."""
    if not session_id:
        raise UploadError("Uploads need a session.")
    doc_id = clear_of_shared(store, slugify(Path(filename).stem))
    try:
        # PDF parsing is CPU-bound: off the event loop, so other chats keep streaming.
        with time_step("knowledge", "extract_text"):
            text = await asyncio.to_thread(extract_text, filename, data)
        transcribed = False
    except NoTextLayerError as exc:
        text = await _transcribe(session_id, doc_id, data, exc.pages, store, provider, charge_transcription)
        transcribed = True
    _, text = as_note(filename, text)
    result = await store_private_note(session_id, doc_id, text, store, provider, charge)
    return {**result, "transcribed": True} if transcribed else result


async def _transcribe(
    session_id: str,
    doc_id: str,
    data: bytes,
    pages: int,
    store: VectorStore,
    provider: LLMProvider,
    charge: Callable[[], None] | None,
) -> str:
    """A scanned PDF's text, read by the model. Everything that would refuse
    the note anyway is checked first, so no model call is spent on it."""
    if pages > MAX_SCANNED_PAGES:
        raise UploadError(
            f"That PDF is scanned (it has no text layer), and a scanned PDF can have up to "
            f"{MAX_SCANNED_PAGES} pages; this one has {pages}. Split it into smaller files."
        )
    _own_notes(store, session_id, doc_id)
    if store.private_count() >= get_settings().max_private_chunks:
        raise PrivateNotesFullError(
            "The assistant is holding as many private notes as it has room for. "
            "Try again later, once older notes have expired."
        )
    if charge is not None:
        charge()
    with time_step("inference", "transcribe_pdf", pages=pages):
        text, finish_reason = await provider.transcribe_pdf(data)
    text = text.strip()
    if finish_reason != "stop" or not text:
        raise UploadError(_TRANSCRIBE_REFUSALS.get(finish_reason, "No text could be read from that scanned PDF."))
    if len(text) > MAX_UPLOAD_CHARS:
        raise UploadError(f"That PDF has over {MAX_UPLOAD_CHARS:,} characters of text. Split it into smaller files.")
    return text
