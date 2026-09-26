import contextvars
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.inference.provider import LLMProvider
from app.knowledge.store import VectorStore
from app.knowledge.uploads import UPLOAD_TTL_S
from app.observability import time_step


@dataclass
class RetrievedChunk:
    doc_id: str
    text: str
    score: float


# The session being served, so a search sees the shared notes plus that
# session's private uploads without threading a session id through the agent
# loop and every tool signature. The intelligence layer sets it per turn.
current_session: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session", default=None
)


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
    settings = get_settings()
    store = store or get_store()
    k = k or settings.retrieval_top_k
    with time_step("inference", "embed_query"):
        [query_vec] = await provider.embed([query], "query")
    with time_step("knowledge", "vector_search", k=k):
        store.purge_uploads(UPLOAD_TTL_S)  # so expired uploads stop showing up on time
        hits = store.search(query_vec, k, owner=current_session.get())
    # Relative, not an absolute cutoff: a terse query's right answer can score
    # as low as an unrelated note does for another query, but it still leads.
    if hits:
        floor = hits[0].score - settings.retrieval_score_margin
        hits = [h for h in hits if h.score >= floor]
    return [RetrievedChunk(h.doc_id, h.text, round(h.score, 4)) for h in hits]
