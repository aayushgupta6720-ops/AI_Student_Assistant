"""CLI: ingest every markdown note in data/notes into the vector store.

    python -m scripts.ingest [--reset]
"""

import argparse
import asyncio

from app.config import get_settings
from app.inference.gemini import get_provider
from app.knowledge.ingest import ingest_dir
from app.knowledge.retrieval import get_store


async def main(reset: bool) -> None:
    settings = get_settings()
    store = get_store()
    if reset:
        store.clear()
    counts = await ingest_dir(settings.notes_dir, store, get_provider())
    for doc_id, n in counts.items():
        print(f"  {doc_id}: {n} chunks")
    print(f"{len(counts)} docs, {store.count()} chunks in {settings.store_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="wipe the store first")
    args = parser.parse_args()
    asyncio.run(main(args.reset))
