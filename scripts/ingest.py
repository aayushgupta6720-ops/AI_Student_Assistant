"""CLI: sync the vector store with the markdown notes in data/notes: new and
changed notes are embedded (unchanged ones are skipped), and notes deleted
from the folder are dropped from the store.

    python -m scripts.ingest
"""

import asyncio

from app.config import get_settings
from app.inference.gemini import get_provider
from app.knowledge.ingest import ingest_dir
from app.knowledge.retrieval import get_store


async def main() -> None:
    settings = get_settings()
    store = get_store()
    counts = await ingest_dir(settings.notes_dir, store, get_provider())
    for doc_id, n in counts.items():
        print(f"  {doc_id}: {n} chunks")
    print(f"{len(counts)} docs, {store.count()} chunks in {settings.store_path}")


if __name__ == "__main__":
    asyncio.run(main())
