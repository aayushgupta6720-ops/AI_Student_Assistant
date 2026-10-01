import math

import pytest

from app.knowledge.retrieval import current_session, identifiers, retrieve
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


def _handbook_store(tmp_path) -> VectorStore:
    """The physics sheet scores 0.9 by meaning for any query; the handbook
    line naming the exam hall scores 0.1, far below the cutoff."""
    store = VectorStore(tmp_path / "s.sqlite")
    store.upsert_doc("physics", ["Bernoulli: pressure falls where a fluid speeds up."], [[0.9, math.sqrt(1 - 0.81)]])
    store.upsert_doc("handbook", ["The final exam is on 28 November in MPSH 2A."], [[0.1, math.sqrt(1 - 0.01)]])
    return store


async def test_a_code_in_the_query_brings_the_passage_that_names_it_first(tmp_path):
    # "MPSH 2A" returned only the physics sheet in the answer eval, every run.
    hits = await retrieve("What's happening in MPSH 2A?", QueryProvider(), _handbook_store(tmp_path), k=4)

    assert [h.doc_id for h in hits] == ["handbook", "physics"]
    assert hits[0].score == pytest.approx(0.1, abs=1e-4)  # still scored by meaning, like any passage


async def test_a_question_without_codes_searches_by_meaning_alone(tmp_path):
    hits = await retrieve("What's happening in the exam hall?", QueryProvider(), _handbook_store(tmp_path), k=4)

    assert [h.doc_id for h in hits] == ["physics"]


async def test_keyword_matches_count_towards_k(tmp_path):
    hits = await retrieve("MPSH 2A", QueryProvider(), _handbook_store(tmp_path), k=1)

    assert [h.doc_id for h in hits] == ["handbook"]


async def test_keyword_matches_only_come_from_notes_the_session_can_see(tmp_path):
    store = _handbook_store(tmp_path)
    store.upsert_doc("timetable", ["Bob's tutorial T07 is on Monday."], [[0.0, 1.0]], owner="bob")

    current_session.set("alice")
    hits = await retrieve("When is T07?", QueryProvider(), store, k=4)

    assert "timetable" not in {h.doc_id for h in hits}


@pytest.mark.parametrize(("query", "terms"), [
    ("What's happening in MPSH 2A?", ["MPSH", "2A"]),
    ("Why does MAT1150 matter for CSC2204?", ["MAT1150", "CSC2204"]),
    ("what does n2 mean in snell's law", ["n2"]),
    ("What is on my reading list?", []),
    ("I want to know", []),  # one capital isn't a code
])
def test_codes_and_symbols_are_what_get_matched_as_keywords(query, terms):
    assert identifiers(query) == terms
