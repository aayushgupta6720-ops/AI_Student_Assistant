from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.inference.provider import LLMProvider
from app.knowledge.store import VectorStore
from app.observability import time_step


@dataclass
class RetrievedChunk:
    doc_id: str
    text: str
    score: float


@lru_cache
def get_store() -> VectorStore:
    settings = get_settings()
    settings.store_path.parent.mkdir(parents=True, exist_ok=True)
    return VectorStore(settings.store_path)


async def retrieve(
    query: str,
    provider: LLMProvider,
    store: VectorStore | None = None,
    k: int | None = None,
) -> list[RetrievedChunk]:
    store = store or get_store()
    k = k or get_settings().retrieval_top_k
    with time_step("inference", "embed_query"):
        [query_vec] = await provider.embed([query], "query")
    with time_step("knowledge", "vector_search", k=k):
        hits = store.search(query_vec, k)
    return [RetrievedChunk(h.doc_id, h.text, round(h.score, 4)) for h in hits]
