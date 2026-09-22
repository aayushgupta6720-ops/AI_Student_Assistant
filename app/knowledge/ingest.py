"""Ingestion pipeline: file -> chunks -> embeddings (via inference) -> store."""

from pathlib import Path

from app.config import get_settings
from app.inference.provider import LLMProvider
from app.knowledge.chunking import chunk_text
from app.knowledge.store import VectorStore
from app.observability import time_step


async def ingest_file(path: Path, store: VectorStore, provider: LLMProvider) -> int:
    settings = get_settings()
    doc_id = path.stem
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text, settings.chunk_max_chars, settings.chunk_overlap_chars)
    if not chunks:
        store.delete_doc(doc_id)
        return 0
    with time_step("inference", "embed_documents", doc_id=doc_id, chunks=len(chunks)):
        embeddings = await provider.embed(chunks, "document")
    with time_step("knowledge", "store_upsert", doc_id=doc_id):
        store.upsert_doc(doc_id, chunks, embeddings)
    return len(chunks)


async def ingest_dir(notes_dir: Path, store: VectorStore, provider: LLMProvider) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(notes_dir.glob("*.md")):
        counts[path.stem] = await ingest_file(path, store, provider)
    return counts
