"""Ingestion pipeline: file -> chunks -> embeddings (via inference) -> store."""

import re
from pathlib import Path

from app.config import get_settings
from app.inference.provider import LLMProvider
from app.knowledge.chunking import Chunk, chunk_markdown
from app.knowledge.store import VectorStore
from app.observability import time_step


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "note"


def _labelled(chunk: Chunk) -> str:
    """The chunk as embedded and stored: headed by the note title and section
    it sits under, so a chunk from the middle of a note still says what it's
    about, both to the embedding and to the model reading search results."""
    if not chunk.headings:
        return chunk.text
    return "\n".join(chunk.headings) + "\n\n" + chunk.text


def chunk_note(text: str) -> list[str]:
    """A markdown note as the chunks to embed and store, each labelled with
    the headings it sits under."""
    settings = get_settings()
    return [
        _labelled(chunk)
        for chunk in chunk_markdown(text, settings.chunk_max_chars, settings.chunk_overlap_chars)
    ]


async def ingest_file(path: Path, store: VectorStore, provider: LLMProvider) -> int:
    doc_id = path.stem
    chunks = chunk_note(path.read_text(encoding="utf-8"))
    if not chunks:
        store.delete_doc(doc_id)
        return 0
    with time_step("inference", "embed_documents", doc_id=doc_id, chunks=len(chunks)):
        embeddings = await provider.embed(chunks, "document")
    with time_step("knowledge", "store_upsert", doc_id=doc_id):
        store.upsert_doc(doc_id, chunks, embeddings)
    return len(chunks)


async def ingest_dir(notes_dir: Path, store: VectorStore, provider: LLMProvider) -> dict[str, int]:
    """Make the store mirror notes_dir: (re-)ingest every note in it, then drop
    notes deleted from it. Pruning runs last, so a failure partway through
    leaves every note searchable, some still with their previous chunks."""
    counts: dict[str, int] = {}
    for path in sorted(notes_dir.glob("*.md")):
        counts[path.stem] = await ingest_file(path, store, provider)
    # A missing directory is more likely a misconfigured path than a user who
    # deleted every note, so don't let it wipe the index. list_docs() with no
    # owner is the shared notes only, so session uploads are never pruned.
    if notes_dir.is_dir():
        for doc in store.list_docs():
            if doc["doc_id"] not in counts:
                store.delete_doc(doc["doc_id"])
    return counts
