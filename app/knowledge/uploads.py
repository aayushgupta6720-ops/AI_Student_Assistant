"""Private notes: uploaded files, and notes saved from the chat with
save_note. Each is chunked, embedded and stored under its session, where only
that session's searches can see it (see store.py). They expire with the
session, or after UPLOAD_TTL_S, whichever comes first."""

import asyncio
import io
from pathlib import Path
from typing import Callable

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import get_settings
from app.inference.provider import LLMProvider
from app.knowledge.ingest import chunk_note, slugify
from app.knowledge.store import VectorStore
from app.observability import time_step

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
# The byte cap alone doesn't bound the work: a compressed PDF well under it
# can hold megabytes of text, and every ~800 chars is another chunk to embed.
# 200,000 chars is about 100 pages of notes, and a few embed requests.
MAX_UPLOAD_CHARS = 200_000
MAX_UPLOADS_PER_SESSION = 10
UPLOAD_TTL_S = 24 * 3600
ALLOWED_SUFFIXES = (".md", ".txt", ".pdf")


class UploadError(ValueError):
    """An upload we refuse, with a message fit to show the user."""


class PrivateNotesFullError(UploadError):
    """The app holds as many private notes as it has memory for. Nothing is
    wrong with the upload; it can work once older notes expire."""


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadError(f"Only {', '.join(ALLOWED_SUFFIXES)} files can be uploaded.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"That file is over the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    if suffix == ".pdf":
        pages: list[str] = []
        length = 0
        try:
            for page in PdfReader(io.BytesIO(data)).pages:
                pages.append((page.extract_text() or "").strip())
                length += len(pages[-1])
                if length > MAX_UPLOAD_CHARS:
                    break  # already too long; don't extract the rest
        except (PdfReadError, ValueError) as exc:
            raise UploadError("That PDF couldn't be read.") from exc
        text = "\n\n".join(pages)
        if not text.strip():
            raise UploadError("No text found in that PDF. Scanned PDFs without a text layer aren't supported.")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadError("That file isn't UTF-8 text.") from exc
        if not text.strip():
            raise UploadError("That file is empty.")

    text = text.strip()
    if len(text) > MAX_UPLOAD_CHARS:
        raise UploadError(f"That file has over {MAX_UPLOAD_CHARS:,} characters of text. Split it into smaller files.")
    return text


def as_note(filename: str, text: str) -> tuple[str, str]:
    """(doc_id, markdown) for an uploaded file. The note gets a "# Title"
    first line if it lacks one, so the heading-aware chunker labels every
    chunk with the note it came from."""
    stem = Path(filename).stem
    if not text.startswith("# "):
        title = stem.replace("_", " ").replace("-", " ").strip() or "Untitled"
        text = f"# {title}\n\n{text}"
    return slugify(stem), text


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
    embedded, and may raise to refuse it (the API's per-visitor budget)."""
    if not session_id:
        raise UploadError("Private notes need a session.")

    store.purge_uploads(UPLOAD_TTL_S)
    own = {d["doc_id"]: d["chunks"] for d in store.list_docs(owner=session_id) if d["uploaded"]}
    if doc_id not in own and len(own) >= MAX_UPLOADS_PER_SESSION:
        raise UploadError(
            f"You can have up to {MAX_UPLOADS_PER_SESSION} private notes per chat "
            "(uploads and saved notes); remove one first."
        )

    chunks = chunk_note(text)
    # A visitor can start any number of sessions, so the per-session cap
    # doesn't bound memory; this one does. Replacing a note frees its chunks.
    if store.private_count() - own.get(doc_id, 0) + len(chunks) > get_settings().max_private_chunks:
        raise PrivateNotesFullError(
            "The assistant is holding as many private notes as it has room for. "
            "Try again later, once older notes have expired."
        )
    if charge is not None:
        charge(len(chunks))
    with time_step("inference", "embed_documents", doc_id=doc_id, chunks=len(chunks)):
        embeddings = await provider.embed(chunks, "document")
    with time_step("knowledge", "store_upsert", doc_id=doc_id):
        store.upsert_doc(doc_id, chunks, embeddings, owner=session_id)
    return {"doc_id": doc_id, "chunks": len(chunks)}


async def ingest_upload(
    session_id: str,
    filename: str,
    data: bytes,
    store: VectorStore,
    provider: LLMProvider,
    charge: Callable[[int], None] | None = None,
) -> dict:
    """Store an uploaded file as one of `session_id`'s private notes.
    Re-uploading a file with the same name replaces the earlier version."""
    if not session_id:
        raise UploadError("Uploads need a session.")
    # PDF parsing is CPU-bound: off the event loop, so other chats keep streaming.
    text = await asyncio.to_thread(extract_text, filename, data)
    doc_id, text = as_note(filename, text)
    return await store_private_note(session_id, doc_id, text, store, provider, charge)
