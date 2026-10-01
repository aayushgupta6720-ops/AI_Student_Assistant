import sqlite3
import time

import pytest

from app.knowledge.store import VectorStore


def test_upsert_search_delete(tmp_path):
    store = VectorStore(tmp_path / "s.sqlite")
    store.upsert_doc("a", ["a0", "a1"], [[1, 0, 0], [0, 1, 0]])
    store.upsert_doc("b", ["b0"], [[0, 0, 1]])
    assert store.count() == 3

    hits = store.search([0.9, 0.1, 0], k=2)
    assert [h.text for h in hits] == ["a0", "a1"]
    assert hits[0].score > hits[1].score

    store.upsert_doc("a", ["a-new"], [[0, 0, 1]])  # replaces both old chunks
    assert store.count() == 2
    assert store.list_docs() == [
        {"doc_id": "a", "chunks": 1, "uploaded": False},
        {"doc_id": "b", "chunks": 1, "uploaded": False},
    ]

    store.delete_doc("b")
    assert [h.text for h in store.search([0, 0, 1], k=5)] == ["a-new"]


def test_empty_store(tmp_path):
    assert VectorStore(tmp_path / "e.sqlite").search([1, 2, 3], k=3) == []


@pytest.fixture
def store(tmp_path):
    """A shared note plus one upload each for sessions alice and bob."""
    s = VectorStore(tmp_path / "s.sqlite")
    s.upsert_doc("shared", ["shared note"], [[1, 0, 0]])
    s.upsert_doc("lecture", ["alice's lecture"], [[1, 0.1, 0]], owner="alice")
    s.upsert_doc("lecture", ["bob's lecture"], [[1, 0.2, 0]], owner="bob")
    return s


def test_a_search_sees_shared_notes_and_only_its_own_uploads(store):
    assert {h.text for h in store.search([1, 0, 0], k=10, owner="alice")} == {"shared note", "alice's lecture"}
    assert {h.text for h in store.search([1, 0, 0], k=10, owner="bob")} == {"shared note", "bob's lecture"}
    assert {h.text for h in store.search([1, 0, 0], k=10)} == {"shared note"}


def test_list_and_count_are_scoped_the_same_way(store):
    assert store.list_docs(owner="alice") == [
        {"doc_id": "shared", "chunks": 1, "uploaded": False},
        {"doc_id": "lecture", "chunks": 1, "uploaded": True},
    ]
    assert store.list_docs() == [{"doc_id": "shared", "chunks": 1, "uploaded": False}]
    assert store.count() == 1 and store.count(owner="alice") == 2


def test_a_session_can_only_delete_its_own_uploads(store):
    assert not store.delete_doc("shared", owner="alice")  # not alice's
    assert store.delete_doc("lecture", owner="alice")
    assert {h.text for h in store.search([1, 0, 0], k=10, owner="bob")} == {"shared note", "bob's lecture"}


def test_delete_owner_removes_one_sessions_uploads_and_refuses_the_shared_notes(store):
    assert store.delete_owner("alice") == 1
    assert store.list_docs(owner="alice") == [{"doc_id": "shared", "chunks": 1, "uploaded": False}]
    with pytest.raises(ValueError):
        store.delete_owner("")


def test_purge_removes_old_uploads_but_never_shared_notes(store, monkeypatch):
    later = time.time() + 25 * 3600
    monkeypatch.setattr(time, "time", lambda: later)

    assert store.purge_uploads(older_than_s=24 * 3600) == 2
    assert store.list_docs(owner="alice") == [{"doc_id": "shared", "chunks": 1, "uploaded": False}]


def test_a_store_in_the_old_layout_is_rebuilt_empty(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chunks (doc_id TEXT, chunk_index INTEGER, text TEXT, embedding BLOB)")
    conn.execute("INSERT INTO chunks VALUES ('a', 0, 'old', x'00')")
    conn.commit()
    conn.close()

    store = VectorStore(path)

    # empty, so startup re-ingests data/notes into the new layout
    assert store.count() == 0
    store.upsert_doc("a", ["new"], [[1.0]], owner="s")
    assert store.list_docs(owner="s") == [{"doc_id": "a", "chunks": 1, "uploaded": True}]


def test_a_notes_text_is_scoped_like_its_chunks(store):
    store.upsert_doc("shared", ["shared note"], [[1, 0, 0]], source="the shared note")
    store.upsert_doc("lecture", ["alice's lecture"], [[1, 0.1, 0]], owner="alice", source="alice's whole lecture")

    assert store.note_text("lecture", owner="alice") == "alice's whole lecture"
    assert store.note_text("lecture", owner="bob") is None  # bob's lecture was stored without its text
    assert store.note_text("lecture", owner="carol") is None and store.note_text("lecture") is None
    assert store.note_text("shared", owner="carol") == "the shared note"


@pytest.mark.parametrize("remove", [
    lambda s: s.delete_doc("lecture", owner="alice"),
    lambda s: s.delete_owner("alice"),
    lambda s: s.purge_stale_private("another-model@768"),
    lambda s: s.upsert_doc("lecture", ["replaced"], [[1, 0, 0]], owner="alice"),  # without a new text
], ids=["delete_doc", "delete_owner", "purge_stale_private", "upsert"])
def test_a_notes_text_goes_when_its_chunks_do(store, remove):
    # Otherwise read_note could still read a note that's been deleted or has expired.
    store.upsert_doc("lecture", ["alice's lecture"], [[1, 0.1, 0]], owner="alice", source="alice's whole lecture")
    remove(store)
    assert store.note_text("lecture", owner="alice") is None


def test_purged_uploads_lose_their_text_too(store, monkeypatch):
    store.upsert_doc("lecture", ["alice's lecture"], [[1, 0.1, 0]], owner="alice", source="alice's whole lecture")
    store.upsert_doc("shared", ["shared note"], [[1, 0, 0]], source="the shared note")
    later = time.time() + 25 * 3600
    monkeypatch.setattr(time, "time", lambda: later)

    store.purge_uploads(older_than_s=24 * 3600)

    assert store.note_text("lecture", owner="alice") is None
    assert store.note_text("shared") == "the shared note"


@pytest.fixture
def coded(tmp_path):
    s = VectorStore(tmp_path / "k.sqlite")
    s.upsert_doc("handbook", ["The final exam is in MPSH 2A.", "Tutorial T07 meets on Thursday."], [[1, 0], [0, 1]],
                 owner="alice")
    s.upsert_doc("timetable", ["Bob's T07 is on Monday."], [[1, 0]], owner="bob")
    s.upsert_doc("shared", ["Exams are held in MPSH halls."], [[1, 0]])
    return s


def test_keyword_search_finds_a_code_in_the_notes_a_session_can_see(coded):
    assert [h.text for h in coded.keyword_search(["T07"], [1, 0], owner="alice")] == ["Tutorial T07 meets on Thursday."]
    assert [h.text for h in coded.keyword_search(["T07"], [1, 0], owner="carol")] == []
    assert {h.doc_id for h in coded.keyword_search(["MPSH"], [1, 0], owner="carol")} == {"shared"}


def test_keyword_search_ignores_a_term_too_common_to_single_out_a_passage(tmp_path):
    s = VectorStore(tmp_path / "k.sqlite")
    s.upsert_doc("bio", [f"ATP fact {i}" for i in range(4)] + ["Only here: p53."], [[1, 0]] * 5)

    assert s.keyword_search(["ATP"], [1, 0], max_matches=3) == []
    assert [h.text for h in s.keyword_search(["ATP", "p53"], [1, 0], max_matches=3)] == ["Only here: p53."]


@pytest.mark.parametrize("remove", [
    lambda s: s.delete_doc("handbook", owner="alice"),
    lambda s: s.delete_owner("alice"),
    lambda s: s.upsert_doc("handbook", ["Rewritten without the code."], [[1, 0]], owner="alice"),
    lambda s: s.clear(),
], ids=["delete_doc", "delete_owner", "upsert", "clear"])
def test_the_keyword_index_forgets_removed_chunks(coded, remove):
    remove(coded)
    assert coded.keyword_search(["T07"], [1, 0], owner="alice") == []
    # Searches join back to the chunks, which hides a stale index entry, but it
    # would still count towards a term's matches. FTS5 checks the index itself.
    coded._conn.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES ('integrity-check', 1)")  # 1: against the chunks


def test_without_fts5_search_works_by_meaning_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(VectorStore, "_create_keyword_index", lambda self: False)
    s = VectorStore(tmp_path / "k.sqlite")
    s.upsert_doc("handbook", ["The final exam is in MPSH 2A."], [[1, 0]])

    assert not s.keyword_search_enabled and s.keyword_search(["MPSH"], [1, 0]) == []
    assert [h.doc_id for h in s.search([1, 0], k=1)] == ["handbook"]
