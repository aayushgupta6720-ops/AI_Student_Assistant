"""Private note uploads: pull the text out of an uploaded file, then chunk,
embed and store it under the uploading session, where only that session's
searches can see it (see store.py). Uploads expire with the session, or
after UPLOAD_TTL_S, whichever comes first."""

import io
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.inference.provider import LLMProvider
from app.knowledge.ingest import chunk_note, slugify
from app.knowledge.store import VectorStore
from app.observability import time_step

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_UPLOADS_PER_SESSION = 10
UPLOAD_TTL_S = 24 * 3600
ALLOWED_SUFFIXES = (".md", ".txt", ".pdf")


class UploadError(ValueError):
    """An upload we refuse, with a message fit to show the user."""


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadError(f"Only {', '.join(ALLOWED_SUFFIXES)} files can be uploaded.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"That file is over the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    if suffix == ".pdf":
        try:
            pages = PdfReader(io.BytesIO(data)).pages
            text = "\n\n".join((page.extract_text() or "").strip() for page in pages)
        except (PdfReadError, ValueError) as exc:
            raise UploadError("That PDF couldn't be read.") from exc
        if not text.strip():
            raise UploadError("No text found in that PDF. Scanned PDFs without a text layer aren't supported.")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadError("That file isn't UTF-8 text.") from exc
        if not text.strip():
            raise UploadError("That file is empty.")
    return text.strip()


def as_note(filename: str, text: str) -> tuple[str, str]:
    """(doc_id, markdown) for an uploaded file. The note gets a "# Title"
    first line if it lacks one, so the heading-aware chunker labels every
    chunk with the note it came from."""
    stem = Path(filename).stem
    if not text.startswith("# "):
        title = stem.replace("_", " ").replace("-", " ").strip() or "Untitled"
        text = f"# {title}\n\n{text}"
    return slugify(stem), text


async def ingest_upload(
    session_id: str, filename: str, data: bytes, store: VectorStore, provider: LLMProvider
) -> dict:
    """Store an uploaded file as one of `session_id`'s private notes.
    Re-uploading a file with the same name replaces the earlier version."""
    if not session_id:
        raise UploadError("Uploads need a session.")
    doc_id, text = as_note(filename, extract_text(filename, data))

    store.purge_uploads(UPLOAD_TTL_S)
    own = {d["doc_id"] for d in store.list_docs(owner=session_id) if d["uploaded"]}
    if doc_id not in own and len(own) >= MAX_UPLOADS_PER_SESSION:
        raise UploadError(f"You can have up to {MAX_UPLOADS_PER_SESSION} uploads per chat; remove one first.")

    chunks = chunk_note(text)
    with time_step("inference", "embed_documents", doc_id=doc_id, chunks=len(chunks)):
        embeddings = await provider.embed(chunks, "document")
    with time_step("knowledge", "store_upsert", doc_id=doc_id):
        store.upsert_doc(doc_id, chunks, embeddings, owner=session_id)
    return {"doc_id": doc_id, "chunks": len(chunks)}
