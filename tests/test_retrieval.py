import math

from app.knowledge.retrieval import retrieve
from app.knowledge.store import VectorStore


class QueryProvider:
    """Embeds every query as [1, 0], so a stored [cos, sin] vector scores cos."""

    async def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _store(tmp_path, scores: dict[str, float]) -> VectorStore:
    store = VectorStore(tmp_path / "s.sqlite")
    for doc_id, score in scores.items():
        store.upsert_doc(doc_id, [doc_id], [[score, math.sqrt(1 - score**2)]])
    return store


async def test_passages_far_below_the_best_match_are_dropped(tmp_path):
    # Real scores for "What's on my reading list?": the right note 0.74, the
    # rest 0.56-0.58, and every one of them used to be cited as a source.
    store = _store(tmp_path, {"reading-list": 0.74, "trip-and-books": 0.70, "project-plan": 0.58, "pasta": 0.56})

    hits = await retrieve("reading list", QueryProvider(), store, k=4)

    assert [h.doc_id for h in hits] == ["reading-list", "trip-and-books"]


async def test_the_best_match_is_kept_however_weak(tmp_path):
    # A terse query's right answer can score like an unrelated note, so there's
    # no absolute cutoff: the model decides whether a weak best match is relevant.
    store = _store(tmp_path, {"travel": 0.559, "reading": 0.45})

    hits = await retrieve("headphones", QueryProvider(), store, k=4)

    assert [h.doc_id for h in hits] == ["travel"]
